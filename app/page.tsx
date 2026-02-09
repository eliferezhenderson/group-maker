"use client";

import React, { useMemo, useState } from "react";

type Level = "bachelor" | "master";
type Gender = "f" | "m" | "x" | "na"; // x = nonbinary/other, na = unspecified
type Student = {
  id: string;
  name: string;
  gender: Gender;
  level: Level | "na";
  exchange: boolean | "na";
};

type Group = Student[];

function uidFromName(name: string) {
  return name.trim().toLowerCase().replace(/\s+/g, " ");
}

/**
 * Parse CSV-ish lines.
 * Format (one per line):
 * name,gender,level,exchange
 * reinforces a simple workflow for teachers
 *
 * Examples:
 * Ada Lovelace,f,master,false
 * Sam Lee,m,bachelor,true
 * Pat Kim,x,master,false
 * Alex Doe,na,na,na
 */
function parseStudents(input: string): Student[] {
  const lines = input
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));

  const out: Student[] = [];
  for (const line of lines) {
    const parts = line.split(",").map((p) => p.trim());
    const name = parts[0] ?? "";
    if (!name) continue;

    const genderRaw = (parts[1] ?? "na").toLowerCase();
    const levelRaw = (parts[2] ?? "na").toLowerCase();
    const exchangeRaw = (parts[3] ?? "na").toLowerCase();

    const gender: Gender =
      genderRaw === "f" || genderRaw === "m" || genderRaw === "x" || genderRaw === "na"
        ? (genderRaw as Gender)
        : "na";

    const level: Student["level"] =
      levelRaw === "bachelor" || levelRaw === "master" || levelRaw === "na"
        ? (levelRaw as any)
        : "na";

    const exchange: Student["exchange"] =
      exchangeRaw === "true" ? true : exchangeRaw === "false" ? false : "na";

    out.push({
      id: uidFromName(name),
      name,
      gender,
      level,
      exchange,
    });
  }

  // Deduplicate by id, keep first occurrence
  const seen = new Set<string>();
  return out.filter((s) => (seen.has(s.id) ? false : (seen.add(s.id), true)));
}

/** Convert prior groups -> pair set (unordered) to avoid repeats */
function buildForbiddenPairs(previousGroups: Group[]): Set<string> {
  const set = new Set<string>();
  for (const g of previousGroups) {
    for (let i = 0; i < g.length; i++) {
      for (let j = i + 1; j < g.length; j++) {
        const a = g[i].id;
        const b = g[j].id;
        const key = a < b ? `${a}||${b}` : `${b}||${a}`;
        set.add(key);
      }
    }
  }
  return set;
}

function shuffle<T>(arr: T[], rng = Math.random): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Build groups of size groupSize. Last group may be smaller unless "padLast" is on.
 */
function chunkGroups(students: Student[], groupSize: number): Group[] {
  const groups: Group[] = [];
  for (let i = 0; i < students.length; i += groupSize) {
    groups.push(students.slice(i, i + groupSize));
  }
  return groups;
}

type Weights = {
  forbidPair: number; // huge
  genderImbalance: number;
  levelImbalance: number;
  exchangeImbalance: number;
  sizeImbalance: number;
};

function countBy<T extends string>(group: Group, keyFn: (s: Student) => T): Record<T, number> {
  const r = {} as Record<T, number>;
  for (const s of group) {
    const k = keyFn(s);
    r[k] = (r[k] ?? 0) + 1;
  }
  return r;
}

function groupScore(group: Group, forbiddenPairs: Set<string>, targetSize: number, w: Weights): number {
  let score = 0;

  // Pair repeats (hard penalty)
  for (let i = 0; i < group.length; i++) {
    for (let j = i + 1; j < group.length; j++) {
      const a = group[i].id;
      const b = group[j].id;
      const key = a < b ? `${a}||${b}` : `${b}||${a}`;
      if (forbiddenPairs.has(key)) score += w.forbidPair;
    }
  }

  // Size penalty (keep groups near target size; last group can differ)
  score += w.sizeImbalance * Math.abs(group.length - targetSize);

  // Attribute balance heuristics:
  // We treat "na" as neutral/unknown, so it doesn't create strong penalties.
  const genders = countBy(group, (s) => s.gender);
  const levels = countBy(group, (s) => (s.level === "na" ? "na" : s.level));
  const exchange = countBy(group, (s) => (s.exchange === "na" ? "na" : s.exchange ? "true" : "false"));

  // Gender: penalize deviation from as-even-as-possible among known genders.
  // For small groups, "balanced" means no single known gender dominates too hard.
  const knownGenderCount = (genders.f ?? 0) + (genders.m ?? 0) + (genders.x ?? 0);
  if (knownGenderCount >= 2) {
    const vals = [genders.f ?? 0, genders.m ?? 0, genders.x ?? 0].filter((v) => v > 0 || true);
    const max = Math.max(...vals);
    const min = Math.min(...vals);
    score += w.genderImbalance * (max - min);
  }

  // Level: bachelor/master should be mixed if possible
  const knownLevelCount = (levels.bachelor ?? 0) + (levels.master ?? 0);
  if (knownLevelCount >= 2) {
    score += w.levelImbalance * Math.abs((levels.bachelor ?? 0) - (levels.master ?? 0));
  }

  // Exchange: mix true/false if possible
  const knownExCount = (exchange.true ?? 0) + (exchange.false ?? 0);
  if (knownExCount >= 2) {
    score += w.exchangeImbalance * Math.abs((exchange.true ?? 0) - (exchange.false ?? 0));
  }

  return score;
}

function totalScore(groups: Group[], forbiddenPairs: Set<string>, targetSize: number, w: Weights): number {
  return groups.reduce((acc, g) => acc + groupScore(g, forbiddenPairs, targetSize, w), 0);
}

/**
 * Optimize group assignment by random swapping.
 * - Start with random shuffle.
 * - Swap two students between random groups.
 * - Accept if better, or sometimes if worse (annealing).
 */
function generateGroupsOptimized(
  students: Student[],
  groupSize: number,
  forbiddenPairs: Set<string>,
  attempts: number,
  w: Weights
): { groups: Group[]; score: number } {
  if (groupSize < 2) throw new Error("groupSize must be >= 2");
  if (students.length === 0) return { groups: [], score: 0 };

  const targetSize = groupSize;

  // initial
  let bestGroups = chunkGroups(shuffle(students), groupSize);
  let bestScore = totalScore(bestGroups, forbiddenPairs, targetSize, w);

  // current state
  let curGroups = bestGroups.map((g) => [...g]);
  let curScore = bestScore;

  // annealing temp schedule
  const startTemp = 3.0;
  const endTemp = 0.2;

  const groupCount = curGroups.length;

  function randomIndex(n: number) {
    return Math.floor(Math.random() * n);
  }

  for (let t = 0; t < attempts; t++) {
    const temp = startTemp + (endTemp - startTemp) * (t / Math.max(1, attempts - 1));

    // pick two different groups
    const gi = randomIndex(groupCount);
    let gj = randomIndex(groupCount);
    if (groupCount > 1) {
      while (gj === gi) gj = randomIndex(groupCount);
    }

    const g1 = curGroups[gi];
    const g2 = curGroups[gj];
    if (g1.length === 0 || g2.length === 0) continue;

    const i = randomIndex(g1.length);
    const j = randomIndex(g2.length);

    // swap
    const a = g1[i];
    const b = g2[j];
    g1[i] = b;
    g2[j] = a;

    const newScore = totalScore(curGroups, forbiddenPairs, targetSize, w);
    const delta = newScore - curScore;

    const saysYes = delta <= 0 || Math.random() < Math.exp(-delta / temp);

    if (saysYes) {
      curScore = newScore;
      if (curScore < bestScore) {
        bestScore = curScore;
        bestGroups = curGroups.map((g) => [...g]);
      }
    } else {
      // revert
      g1[i] = a;
      g2[j] = b;
    }

    // early exit if perfect (or close enough)
    if (bestScore === 0) break;
  }

  return { groups: bestGroups, score: bestScore };
}

function groupsToText(groups: Group[]) {
  return groups
    .map((g, idx) => {
      const names = g.map((s) => s.name).join(", ");
      return `Group ${idx + 1}: ${names}`;
    })
    .join("\n");
}

export default function Page() {
  const [raw, setRaw] = useState<string>(
    [
      "# name,gender,level,exchange",
      "Ada Lovelace,f,master,false",
      "Sam Lee,m,bachelor,true",
      "Pat Kim,x,master,false",
      "Alex Doe,na,na,na",
      "Mehmet Yilmaz,m,master,true",
      "Elena Rossi,f,bachelor,false",
      "Noah Chen,m,bachelor,false",
      "Mina Park,f,master,true",
      "Taylor Singh,x,bachelor,false",
    ].join("\n")
  );

  const [groupSize, setGroupSize] = useState<number>(3);
  const [attempts, setAttempts] = useState<number>(12000);

  // “history” of previous groupings
  const [previous, setPrevious] = useState<Group[]>([]);
  const forbiddenPairs = useMemo(() => buildForbiddenPairs(previous), [previous]);

  const students = useMemo(() => parseStudents(raw), [raw]);

  const weights: Weights = useMemo(
    () => ({
      forbidPair: 250, // make this big to strongly avoid repeats
      genderImbalance: 6,
      levelImbalance: 4,
      exchangeImbalance: 4,
      sizeImbalance: 1,
    }),
    []
  );

  const [groups, setGroups] = useState<Group[]>([]);
  const [score, setScore] = useState<number>(0);
  const [status, setStatus] = useState<string>("");

  function makeGroups() {
    try {
      setStatus("Generating…");
      const { groups: g, score: sc } = generateGroupsOptimized(
        students,
        groupSize,
        forbiddenPairs,
        attempts,
        weights
      );
      setGroups(g);
      setScore(sc);
      setStatus(sc === 0 ? "Done (perfect constraints match)." : "Done (best effort).");
    } catch (e: any) {
      setStatus(e?.message ?? "Error");
    }
  }

  function acceptAsPrevious() {
    // Add the current groups into history so we avoid repeating these pairs next time
    if (groups.length === 0) return;
    const flattened = groups.flat();
    // sanity: only store if it matches current roster (optional)
    setPrevious((p) => [...p, ...groups.map((g) => [...g])]);
    setStatus("Saved as previous groups (pair-avoidance updated).");
    // optionally clear current
    // setGroups([]);
  }

  function clearPrevious() {
    setPrevious([]);
    setStatus("Cleared previous groups history.");
  }

  const csvTemplate = `# name,gender,level,exchange
# gender: f | m | x | na
# level: bachelor | master | na
# exchange: true | false | na
`;

  return (
    <div className="min-h-screen p-6 md:p-10 bg-neutral-50 text-neutral-900">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="space-y-2">
          <h1 className="text-2xl md:text-3xl font-semibold">Balanced Random Group Generator</h1>
          <p className="text-sm md:text-base text-neutral-600">
            Random groups with balance constraints + pair-repeat avoidance from previous rounds.
          </p>
        </header>

        <section className="grid md:grid-cols-2 gap-4">
          <div className="bg-white rounded-2xl shadow-sm p-4 space-y-3 border border-neutral-200">
            <div className="flex items-center justify-between gap-3">
              <h2 className="font-medium">Students</h2>
              <button
                className="text-xs px-3 py-1 rounded-full border border-neutral-300 hover:bg-neutral-100"
                onClick={() => setRaw(csvTemplate + raw)}
                title="Adds a short legend to the top"
              >
                Add legend
              </button>
            </div>
            <textarea
              className="w-full h-80 p-3 rounded-xl border border-neutral-300 font-mono text-xs leading-5"
              value={raw}
              onChange={(e) => setRaw(e.target.value)}
            />
            <p className="text-xs text-neutral-500">
              One student per line: <span className="font-mono">name,gender,level,exchange</span>
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-sm p-4 space-y-4 border border-neutral-200">
            <h2 className="font-medium">Settings</h2>

            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1">
                <div className="text-xs text-neutral-600">Group size</div>
                <input
                  className="w-full rounded-xl border border-neutral-300 px-3 py-2"
                  type="number"
                  min={2}
                  value={groupSize}
                  onChange={(e) => setGroupSize(Math.max(2, Number(e.target.value)))}
                />
              </label>

              <label className="space-y-1">
                <div className="text-xs text-neutral-600">Search iterations</div>
                <input
                  className="w-full rounded-xl border border-neutral-300 px-3 py-2"
                  type="number"
                  min={1000}
                  step={1000}
                  value={attempts}
                  onChange={(e) => setAttempts(Math.max(1000, Number(e.target.value)))}
                />
              </label>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                className="px-4 py-2 rounded-xl bg-neutral-900 text-white hover:opacity-90"
                onClick={makeGroups}
              >
                Generate
              </button>
              <button
                className="px-4 py-2 rounded-xl border border-neutral-300 hover:bg-neutral-100"
                onClick={makeGroups}
              >
                Reshuffle
              </button>
              <button
                className="px-4 py-2 rounded-xl border border-neutral-300 hover:bg-neutral-100"
                onClick={acceptAsPrevious}
                disabled={groups.length === 0}
                title="Stores these groups as previous so we avoid repeating pairs next time."
              >
                Save as “previous”
              </button>
              <button
                className="px-4 py-2 rounded-xl border border-neutral-300 hover:bg-neutral-100"
                onClick={clearPrevious}
                disabled={previous.length === 0}
              >
                Clear history
              </button>
            </div>

            <div className="text-xs text-neutral-600 space-y-1">
              <div>
                Students loaded: <span className="font-medium">{students.length}</span>
              </div>
              <div>
                Previous groups stored: <span className="font-medium">{previous.length}</span>{" "}
                <span className="text-neutral-400">(used to block repeated pairs)</span>
              </div>
              <div>
                Status: <span className="font-medium">{status || "—"}</span>
              </div>
              <div>
                Score:{" "}
                <span className="font-medium">
                  {groups.length ? score.toFixed(2) : "—"}
                </span>{" "}
                <span className="text-neutral-400">(lower is better; 0 means no penalties)</span>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-white rounded-2xl shadow-sm p-4 border border-neutral-200 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-medium">Output</h2>
            <button
              className="text-xs px-3 py-1 rounded-full border border-neutral-300 hover:bg-neutral-100"
              onClick={() => {
                const txt = groupsToText(groups);
                navigator.clipboard.writeText(txt);
                setStatus("Copied groups to clipboard.");
              }}
              disabled={groups.length === 0}
            >
              Copy
            </button>
          </div>

          {groups.length === 0 ? (
            <p className="text-sm text-neutral-500">Generate groups to see results.</p>
          ) : (
            <div className="grid md:grid-cols-2 gap-3">
              {groups.map((g, idx) => {
                const gCounts = {
                  f: g.filter((s) => s.gender === "f").length,
                  m: g.filter((s) => s.gender === "m").length,
                  x: g.filter((s) => s.gender === "x").length,
                  ba: g.filter((s) => s.level === "bachelor").length,
                  ma: g.filter((s) => s.level === "master").length,
                  ex: g.filter((s) => s.exchange === true).length,
                };

                return (
                  <div key={idx} className="rounded-xl border border-neutral-200 p-3">
                    <div className="flex items-center justify-between">
                      <div className="font-medium">Group {idx + 1}</div>
                      <div className="text-xs text-neutral-500">
                        g:{gCounts.f}/{gCounts.m}/{gCounts.x} · lvl:{gCounts.ba}/{gCounts.ma} · ex:{gCounts.ex}
                      </div>
                    </div>
                    <ul className="mt-2 space-y-1">
                      {g.map((s) => (
                        <li key={s.id} className="text-sm">
                          {s.name}{" "}
                          <span className="text-xs text-neutral-500">
                            ({s.gender}, {s.level}, {String(s.exchange)})
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <footer className="text-xs text-neutral-500">
          Tip: If constraints feel “impossible” (e.g., very uneven gender ratios), the generator will still return the
          best-effort grouping with a non-zero score. Increase “Search iterations” for harder rosters.
        </footer>
      </div>
    </div>
  );
}
