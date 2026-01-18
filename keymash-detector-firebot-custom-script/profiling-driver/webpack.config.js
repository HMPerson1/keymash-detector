module.exports = {
  mode: "development",
  experiments: {
    asyncWebAssembly: true
  },
  devServer: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    }
  }
};
