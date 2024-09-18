const webpack = require('webpack');

module.exports = function override(config) {
  config.resolve.fallback = {
    "fs": false,
    "path": require.resolve("path-browserify"),
    "os": require.resolve("os-browserify/browser"),
    "stream": require.resolve("stream-browserify"),
    "child_process": false,
    "buffer": require.resolve("buffer/"),
    "process": require.resolve("process/browser"),
    "crypto": require.resolve("crypto-browserify")
  };
  config.plugins = (config.plugins || []).concat([
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer']
    }),
  ]);
  return config;
};