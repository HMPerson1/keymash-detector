import type { Firebot } from "@crowbartools/firebot-custom-scripts-types";
import type { EventFilter } from "@crowbartools/firebot-custom-scripts-types/types/modules/event-filter-manager";
import * as packageJson from "../package.json";

interface Params {
  threshold: number;
}

const FILTER_ID = "keymash-detector:has-keymash";

let myParams: Params = { threshold: 1.95 };

const script: Firebot.CustomScript<Params> = {
  getScriptManifest: () => ({
    name: "Keymash Detector",
    description: packageJson.description,
    author: packageJson.author,
    version: packageJson.version,
    startupOnly: true,
    firebotVersion: "5",
  }),
  getDefaultParameters: () => ({
    threshold: {
      type: "number",
      default: 1.95,
      description: "How high the log-likelihood ratio of (keymash : English word) must be in order to pass the filter. Higher values mean fewer messages get through.\n\nRecommended range: 1.5 - 2.0",
      title: "Threshold",
    },
  }),

  stop: undefined,
  async run(runRequest) {
    const { eventFilterManager } = runRequest.modules;
    // can't be at top level b/c commonjs doesn't support top level await,
    // which is required for wasm loading
    const KeymashDetector = await import("../core/pkg");

    const filter: EventFilter = {
      id: FILTER_ID,
      name: "Has Keymash",
      description: "Whether this chat message contains a keymash",
      events: [
        { eventSourceId: "twitch", eventId: "chat-message" },
      ],
      comparisonTypes: ["is"],
      valueType: "preset",
      presetValues: () => [
        { value: "true", display: "True" },
        { value: "false", display: "False" },
      ],
      predicate: (filterSettings, eventData) => {
        const chatMessage = eventData.eventMeta.chatMessage as { parts: Array<any> };
        const maxLlr = chatMessage.parts
          .filter(p => p.type === "text")
          .map(part => KeymashDetector.test_fragment(part.text))
          .reduce((acc, x) => Math.max(acc, x), -Infinity)
        const hasKeymash = maxLlr > myParams.threshold;
        return filterSettings.value === String(hasKeymash);
      }
    };

    myParams.threshold = runRequest.parameters.threshold;
    eventFilterManager.registerFilter(filter);

    this.stop = () => eventFilterManager.unregisterFilter(FILTER_ID);
  },
  parametersUpdated(parameters) {
    myParams.threshold = parameters.threshold
  },
};

export default script;
