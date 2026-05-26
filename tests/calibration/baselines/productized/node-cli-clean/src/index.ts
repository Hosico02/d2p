#!/usr/bin/env node
/** Greeter CLI in TypeScript. */

export function buildMessage(name: string): string {
  return `Hello, ${name}!`;
}

function main(argv: string[]): number {
  const args = argv.slice(2);
  if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
    console.log("usage: greet NAME\n\nPrints a greeting for NAME.");
    return args.length === 0 ? 1 : 0;
  }
  console.log(buildMessage(args[0]));
  return 0;
}

if (require.main === module) {
  process.exit(main(process.argv));
}
