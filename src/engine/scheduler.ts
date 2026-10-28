import type { Operation, SimulationEvent } from "./types";

type Boundary = {
  id: string;
  operation: Operation;
  phase: "start" | "end";
  prerequisites: string[];
  time: number;
  order: number;
};

export function schedule(
  operations: Operation[],
  initial: string[],
): { operations: Operation[]; events: SimulationEvent[] } {
  const result = operations.map((operation) => ({
    ...operation,
    prerequisites: operation.prerequisites.map((dependency) => ({
      ...dependency,
    })),
  }));
  const byId = new Map(result.map((operation) => [operation.id, operation]));
  if (byId.size !== result.length)
    throw new Error("Operation IDs must be unique.");
  const boundaries: Boundary[] = result.flatMap((operation, index) => [
    {
      id: `${operation.id}@start`,
      operation,
      phase: "start" as const,
      prerequisites: operation.prerequisites.map(
        (dependency) => `${dependency.operation}@${dependency.phase}`,
      ),
      time: 0,
      order: index * 2,
    },
    {
      id: `${operation.id}@end`,
      operation,
      phase: "end" as const,
      prerequisites: [`${operation.id}@start`],
      time: 0,
      order: index * 2 + 1,
    },
  ]);
  const boundaryById = new Map(
    boundaries.map((boundary) => [boundary.id, boundary]),
  );
  for (const boundary of boundaries)
    for (const prerequisite of boundary.prerequisites)
      if (!boundaryById.has(prerequisite))
        throw new Error(`Missing prerequisite ${prerequisite}.`);
  for (const operation of result)
    if (!Number.isFinite(operation.duration) || operation.duration < 0)
      throw new Error(`Invalid duration for ${operation.id}.`);
  const scheduled = new Set<string>();
  const ordered: Boundary[] = [];
  while (ordered.length < boundaries.length) {
    const available = boundaries.filter(
      (boundary) =>
        !scheduled.has(boundary.id) &&
        boundary.prerequisites.every((id) => scheduled.has(id)),
    );
    if (!available.length)
      throw new Error("Operation graph contains a dependency cycle.");
    for (const boundary of available) {
      boundary.time =
        Math.max(
          0,
          ...boundary.prerequisites.map((id) => boundaryById.get(id)!.time),
        ) + (boundary.phase === "end" ? boundary.operation.duration : 0);
    }
    available.sort((a, b) => a.time - b.time || a.order - b.order);
    const next = available[0];
    next.operation[next.phase] = next.time;
    scheduled.add(next.id);
    ordered.push(next);
  }
  const previousCompute = new Map<number, Operation>();
  let previousCollective: Operation | undefined;
  for (const operation of result) {
    const previous =
      operation.lane === "collective"
        ? previousCollective
        : previousCompute.get(operation.ranks[0]);
    operation.resourceReadyAt = previous?.end ?? 0;
    const semanticDependencies =
      operation.lane === "collective"
        ? operation.prerequisites.filter(
            (dependency) => dependency.operation !== previous?.id,
          )
        : operation.prerequisites;
    operation.inputReadyAt = Math.max(
      0,
      ...semanticDependencies.map(
        (dependency) => byId.get(dependency.operation)![dependency.phase],
      ),
    );
    const rankCount = Math.max(...result.flatMap((op) => op.ranks)) + 1;
    operation.readyByRank = Array.from({ length: rankCount }, (_, rank) =>
      Math.max(
        0,
        ...semanticDependencies
          .filter((dependency) =>
            byId.get(dependency.operation)!.ranks.includes(rank),
          )
          .map(
            (dependency) => byId.get(dependency.operation)![dependency.phase],
          ),
      ),
    );
    if (operation.lane === "collective") previousCollective = operation;
    else previousCompute.set(operation.ranks[0], operation);
  }
  const events: SimulationEvent[] = [
    {
      index: 0,
      time: 0,
      operation: null,
      phase: "initialize",
      allocated: [...initial],
      released: [],
    },
  ];
  for (const boundary of ordered)
    events.push({
      index: events.length,
      time: boundary.time,
      operation: boundary.operation.id,
      phase: boundary.phase,
      allocated:
        boundary.phase === "start" ? [...boundary.operation.allocate] : [],
      released: boundary.phase === "end" ? [...boundary.operation.release] : [],
    });
  return { operations: result, events };
}
