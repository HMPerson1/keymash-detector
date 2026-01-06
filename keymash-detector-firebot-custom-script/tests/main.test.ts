import { test, expect } from "@jest/globals";
import customScript from "../src/main";
test("main default export is the custom script", () => {
  expect(customScript).not.toBeUndefined();
  expect(customScript.run).not.toBeUndefined();
  expect(customScript.getScriptManifest).not.toBeUndefined();
  expect(customScript.getDefaultParameters).not.toBeUndefined();
});

// lmao that's the only thing we can test b/c running it requires 
// importing the wasm module and jest doesn't like wasm-bindgen output
