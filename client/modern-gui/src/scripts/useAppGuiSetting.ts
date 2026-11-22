import { useEffect, useState } from "react"
import { fetchNoCache } from "../utils/cacheBuster"

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
    const [appGuiSetting, setAppGuiSetting] = useState<AppGuiSetting>({ version: __APP_VERSION__, edition: "Web Version" })
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
                    fetchNoCache('/version'),
                    fetchNoCache('/edition')
                ]);
                const version = (await versionRes.text()).trim();
                const edition = (await editionRes.text()).trim();
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
        const res = await fetchNoCache(url, {
            method: "GET",
        });
        const appSetting = await res.json() as AppGuiSetting;

        setAppGuiSetting({ ...appSetting, version: __APP_VERSION__ });
    }

    return {
        appGuiSetting,
        serverInfo,
    }
}
