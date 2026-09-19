import { expect, it } from "vitest";

import { resolveApiBaseUrl } from "./api";

it("uses the localhost backend during Vite development", () => {
  expect(resolveApiBaseUrl(undefined, true)).toBe("http://127.0.0.1:8000");
});

it("uses same-origin API paths in production", () => {
  expect(resolveApiBaseUrl(undefined, false)).toBe("");
});

it("uses an explicit origin in every mode and removes its trailing slash", () => {
  expect(resolveApiBaseUrl("https://api.example.test/", false)).toBe(
    "https://api.example.test",
  );
  expect(resolveApiBaseUrl("https://api.example.test/", true)).toBe(
    "https://api.example.test",
  );
});
