const path = require("path");
const webpack = require("webpack");
module.exports = {
    entry: "./src/index.ts",
    resolve: {
        extensions: [".ts", ".js"],
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
                        loader: "ts-loader",
                        options: {
                            configFile: "tsconfig.json",
                        },
                    },
                ],
            },
        ],
    },
    output: {
        filename: "index.js",
        path: path.resolve(__dirname, "dist"),
        libraryTarget: "umd",
        globalObject: "typeof self !== 'undefined' ? self : this",
    },

    plugins: [
        new webpack.ProvidePlugin({
            Buffer: ["buffer", "Buffer"],
        }),
        {
            apply(compiler) {
                compiler.hooks.thisCompilation.tap("EmitRootDtsPlugin", (compilation) => {
                    compilation.hooks.processAssets.tap(
                        {
                            name: "EmitRootDtsPlugin",
                            stage: webpack.Compilation.PROCESS_ASSETS_STAGE_ADDITIONS,
                        },
                        () => {
                            compilation.emitAsset(
                                "index.d.ts",
                                new webpack.sources.RawSource('export * from "./src/index";\n')
                            );
                        }
                    );
                });
            },
        },
    ],
    externals: {
        react: "react",
        "react-dom": "reactDOM",
    },
};
