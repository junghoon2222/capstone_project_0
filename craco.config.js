// craco.config.js
const webpack = require('webpack');

module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.resolve.fallback = {
        "fs": false,
        "path": require.resolve("path-browserify"),
        "os": require.resolve("os-browserify/browser"),
        "stream": require.resolve("stream-browserify"),
        "process": require.resolve("process/browser"),
        "child_process": false,
        "crypto": require.resolve("crypto-browserify"),
        "buffer": require.resolve("buffer"),
        "util": require.resolve("util"), // 명시적으로 false로 설정
      };
      webpackConfig.plugins = (webpackConfig.plugins || []).concat([
        new webpack.ProvidePlugin({
          process: 'process/browser',
          Buffer: ['buffer', 'Buffer']
        }),
      ]);
      return webpackConfig;
    }
  }
};
