const path = require("path");
const fs = require("fs");
const { execFileSync } = require("child_process");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");
const webpack = require("webpack");

function resolveAppVersion() {
    const envVer = (process.env.APP_VERSION || process.env.BUILD_NAME || "").trim();
    if (envVer) {
        return envVer;
    }
    const repoRoot = path.resolve(__dirname, "../..");
    const versionFile = path.join(repoRoot, "VERSION");
    let base = "1.0.0";
    try {
        const raw = fs.readFileSync(versionFile, "utf8").trim();
        if (raw) base = raw;
    } catch (_) {}

    try {
        const count = execFileSync("git", ["rev-list", "--count", "HEAD"], {
            cwd: repoRoot,
            encoding: "utf8",
            stdio: ["ignore", "pipe", "ignore"],
        }).trim();
        const shortHash = execFileSync("git", ["rev-parse", "--short=7", "HEAD"], {
            cwd: repoRoot,
            encoding: "utf8",
            stdio: ["ignore", "pipe", "ignore"],
        }).trim();
        if (count && shortHash) {
            return `v${base}-b${count}-${shortHash}`;
        }
        if (shortHash) {
            return `v${base}-${shortHash}`;
        }
    } catch (_) {}
    return `v${base}`;
}

module.exports = {
    mode: "production",
    entry: "./src/index.tsx",
    performance: {
        hints: false,
    },
    resolve: {
        extensions: [".ts", ".tsx", ".js"],
        fallback: {
            buffer: require.resolve("buffer/"),
        },
    },
    module: {
        rules: [
            {
                test: [/\.ts$/, /\.tsx$/],
                use: [
                  {
                      loader: "babel-loader",
                      options: {
                          presets: ["@babel/preset-env", "@babel/preset-react", "@babel/preset-typescript"],
                          plugins: ["@babel/plugin-transform-runtime"],
                      },
                  },
              ],
            },
            {
                test: /\.html$/,
                loader: "html-loader",
            },
            {
                test: /\.css$/,
                use: ["style-loader", { loader: "css-loader", options: { importLoaders: 1 } }, "postcss-loader"],
            },
            { test: /\.json$/, type: "asset/inline" },
        ],
    },
    output: {
        filename: "index.[contenthash:8].js",
        chunkFilename: "[name].[contenthash:8].js",
        path: path.resolve(__dirname, "dist"),
        clean: true,
    },
    plugins: [
        new webpack.DefinePlugin({
            __APP_VERSION__: JSON.stringify(resolveAppVersion()),
        }),
        new webpack.ProvidePlugin({
            Buffer: ["buffer", "Buffer"],
            React: "react",
        }),
        new HtmlWebpackPlugin({
            template: path.resolve(__dirname, "public/index.html"),
            filename: "./index.html",
        }),
        new CopyPlugin({
            patterns: [
                { from: "public/assets", to: "assets", noErrorOnMissing: true }
            ],
        }),
        new CopyPlugin({
            patterns: [
                { from: "public/logo.png", to: "logo.png", noErrorOnMissing: true }
            ],
        }),
        new CopyPlugin({
            patterns: [
                { from: "public/favicon.ico", to: "favicon.ico", noErrorOnMissing: true }
            ],
        })
    ],
};