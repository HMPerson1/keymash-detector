import type { Firebot } from "@crowbartools/firebot-custom-scripts-types";
import type { EventFilter } from "@crowbartools/firebot-custom-scripts-types/types/modules/event-filter-manager";
import type { ReplaceVariable } from "@crowbartools/firebot-custom-scripts-types/types/modules/replace-variable-manager";
import * as packageJson from "../package.json";

interface Params {
  threshold: number;
}

type KeymashDetector = typeof import("../core/pkg");

const REPLVAR_HANDLE = "textKeymashScore";
const FILTER_ID = 'keymash-detector:has-keymash';

const makeReplaceVariable = (KeymashDetector: KeymashDetector): ReplaceVariable => ({
  definition: {
    handle: REPLVAR_HANDLE,
    usage: `${REPLVAR_HANDLE}[text]`,
    description: "Computes the highest keymash score of all words in the input",
    examples: [
      { usage: `${REPLVAR_HANDLE}["some normal english words"]`, description: "-4.999679392483785" },
      { usage: `${REPLVAR_HANDLE}["asdjfkl;"]`, description: "4.296925404801158" },
    ],
    categories: ['text'],
    possibleDataOutput: ['number'],
  },
  evaluator: (_trigger, arg) => KeymashDetector.test_fragment(`${arg}`),
});

const myParams: Params = { threshold: 1.95 };

// captures `myParams`
const makeEventFilter = (KeymashDetector: KeymashDetector): EventFilter => ({
  id: FILTER_ID,
  name: "Has Keymash",
  description: "Whether this chat message contains a keymash",
  events: [
    { eventSourceId: 'twitch', eventId: 'chat-message' },
  ],
  comparisonTypes: ["is"],
  valueType: 'preset',
  // can't be actual primitive booleans because firebot gets confused by non-truthy values
  presetValues: () => [
    { value: 'true', display: "True" },
    { value: 'false', display: "False" },
  ],
  predicate: (filterSettings, eventData) => {
    const chatMessage = eventData.eventMeta.chatMessage as { parts: Array<any>; };
    const maxLlr = chatMessage.parts
      .filter(p => p.type === 'text')
      .map(part => KeymashDetector.test_fragment(part.text))
      .reduce((acc, x) => Math.max(acc, x), -Infinity);
    const hasKeymash = maxLlr > myParams.threshold;
    return filterSettings.value === `${hasKeymash}`;
  },
});

const script: Firebot.CustomScript<Params> = {
  getScriptManifest: () => ({
    name: "Keymash Detector",
    description: packageJson.description,
    website: packageJson.homepage,
    author: packageJson.author,
    version: packageJson.version,
    startupOnly: true,
    firebotVersion: '5',
  }),
  getDefaultParameters: () => ({
    threshold: {
      type: 'number',
      title: "Filter Threshold",
      description: `
How high the log-likelihood ratio of (keymash : English word) must be in order to pass the filter. Higher values mean fewer messages get through.

Only affects the "Has Keymash" event filter.`,
      tip: "Recommended range: 1.5 - 2.0",
      default: 1.95,
    },
  }),

  stop: undefined,
  async run(runRequest) {
    const { eventFilterManager, replaceVariableManager } = runRequest.modules;
    // can't be at top level b/c commonjs doesn't support top level await,
    // which is required for wasm loading
    const KeymashDetector = await import("../core/pkg");

    myParams.threshold = runRequest.parameters.threshold;

    eventFilterManager.registerFilter(makeEventFilter(KeymashDetector));
    replaceVariableManager.registerReplaceVariable(makeReplaceVariable(KeymashDetector));

    this.stop = () => {
      replaceVariableManager.unregisterReplaceVariable(REPLVAR_HANDLE);
      eventFilterManager.unregisterFilter(FILTER_ID);
    };
  },
  parametersUpdated(parameters) {
    myParams.threshold = parameters.threshold
  },
};

export default script;
