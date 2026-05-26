import { describe, it, expect } from "vitest";
import { buildMessage } from "../src/index";

describe("buildMessage", () => {
  it("greets by name", () => {
    expect(buildMessage("World")).toBe("Hello, World!");
  });
});
