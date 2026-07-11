import { useEffect, useState } from "react"

export type AppGuiSetting = {
    version: string,
    edition: string,
}

export type ServerInfo = {
    version: string,
    edition: string,
}

export type AppGuiSettingState = {
    appGuiSetting: AppGuiSetting
    serverInfo: ServerInfo
}

// Custom hook to fetch app GUI settings and server info
export const useAppGuiSetting = (): AppGuiSettingState => {
    // ---------------- State ----------------
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>({ version: "", edition: "" })
    const [serverInfo, setServerInfo] = useState<ServerInfo>({ version: "", edition: "" })

    // ---------------- Hooks ----------------

    // Initial loading of app GUI settings
    useEffect(() => {
        getAppGuiSetting("assets/gui_settings/GUI.json");
    }, [])

    // Initial loading of server version and edition
    useEffect(() => {
        const getServerInfo = async () => {
            try {
                const [versionRes, editionRes] = await Promise.all([
                    fetch('/version'),
                    fetch('/edition')
                ]);
                const version = await versionRes.text();
                const edition = await editionRes.text();
                setServerInfo({ version, edition });
            } catch (e) {
                console.error("Failed to load server info:", e);
            }
        }
        getServerInfo()
    }, [])

    // ---------------- Functions ----------------

    // Fetch app GUI settings from server
    const getAppGuiSetting = async (url: string) => {
        const res = await fetch(`${url}`, {
            method: "GET",
        });
        const appSetting = await res.json() as AppGuiSetting;

        setAppGuiSetting(appSetting);
    }

    return {
        appGuiSetting,
        serverInfo,
    }
}
