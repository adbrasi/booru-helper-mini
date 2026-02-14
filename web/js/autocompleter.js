import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { api } from "../../../scripts/api.js";
import { $el, ComfyDialog } from "../../../scripts/ui.js";
import { TextAreaAutoComplete } from "./common/autocomplete.js";

const id = "booruhelper.AutoCompleter";
const DEFAULT_CUSTOM_WORDS_URL = "https://raw.githubusercontent.com/adbrasi/somethings/refs/heads/main/boorutaags.txt";
const MAX_CUSTOM_WORDS = 40000;
const LARGE_LIST_LINE_THRESHOLD = 50000;
const MAX_ALIASES_PER_WORD = 10;

function parseCSV(csvText) {
	const rows = [];
	const delimiter = ",";
	const quote = '"';
	let currentField = "";
	let inQuotedField = false;

	function pushField() {
		rows[rows.length - 1].push(currentField);
		currentField = "";
		inQuotedField = false;
	}

	rows.push([]);

	for (let i = 0; i < csvText.length; i++) {
		const char = csvText[i];
		const nextChar = csvText[i + 1];
		if (char === "\\" && nextChar === quote) {
			currentField += quote;
			i++;
		}

		if (!inQuotedField) {
			if (char === quote) {
				inQuotedField = true;
			} else if (char === delimiter) {
				pushField();
			} else if (char === "\r" || char === "\n" || i === csvText.length - 1) {
				pushField();
				if (nextChar === "\n") {
					i++;
				}
				rows.push([]);
			} else {
				currentField += char;
			}
		} else {
			if (char === quote && nextChar === quote) {
				currentField += quote;
				i++;
			} else if (char === quote) {
				inQuotedField = false;
			} else if (char === "\r" || char === "\n" || i === csvText.length - 1) {
				const parsed = parseCSV(currentField);
				rows.pop();
				rows.push(...parsed);
				inQuotedField = false;
				currentField = "";
				rows.push([]);
			} else {
				currentField += char;
			}
		}
	}

	if (currentField || csvText[csvText.length - 1] === ",") {
		pushField();
	}

	if (rows[rows.length - 1].length === 0) {
		rows.pop();
	}

	return rows;
}

async function getCustomWords() {
	const resp = await api.fetchApi("/booruhelper/autocomplete", { cache: "no-store" });
	if (resp.status === 200) {
		return await resp.text();
	}
	return undefined;
}

function parseCustomWords(text) {
	const words = {};
	let entries = 0;

	function addWord(wordText, priority, value) {
		if (!wordText) return;
		if (entries >= MAX_CUSTOM_WORDS) return;
		if (words[wordText]) return;
		words[wordText] = { text: wordText, priority, value };
		entries += 1;
	}

	const lineCount = (text.match(/\n/g) || []).length;
	if (lineCount > LARGE_LIST_LINE_THRESHOLD) {
		for (const rawLine of text.split(/\r?\n/)) {
			if (entries >= MAX_CUSTOM_WORDS) break;
			const line = rawLine.trim();
			if (!line) continue;
			const comma = line.indexOf(",");
			let tag = comma === -1 ? line : line.slice(0, comma);
			tag = tag.trim();
			if (tag.startsWith('"') && tag.endsWith('"') && tag.length > 1) {
				tag = tag.slice(1, -1);
			}
			addWord(tag);
		}
		return words;
	}

	for (const row of parseCSV(text)) {
		if (entries >= MAX_CUSTOM_WORDS) break;
		let wordText;
		let priority;
		let value;
		let num;

		switch (row.length) {
			case 0:
				break;
			case 1:
				wordText = row[0];
				addWord(wordText);
				break;
			case 2:
				num = +row[1];
				if (isNaN(num)) {
					wordText = row[0] + "🔄️" + row[1];
					value = row[0];
				} else {
					wordText = row[0];
					priority = num;
				}
				addWord(wordText, priority, value);
				break;
			case 4:
				value = row[0];
				priority = +row[2];
				const aliases = row[3]?.trim();
				if (aliases && aliases !== "null") {
					const split = aliases.split(",");
					for (const alias of split.slice(0, MAX_ALIASES_PER_WORD)) {
						addWord(alias.trim(), priority, value);
					}
				}
				addWord(value, priority, value);
				break;
			default:
				wordText = row[1];
				value = row[0];
				priority = +row[2];
				addWord(wordText, priority, value);
				break;
		}
	}

	return words;
}

async function addCustomWords(text) {
	if (!text) {
		text = await getCustomWords();
	}
	if (text) {
		TextAreaAutoComplete.updateWords("booruhelper.customwords", parseCustomWords(text));
	}
}

class CustomWordsDialog extends ComfyDialog {
	async show() {
		const text = await getCustomWords();
		this.words = $el("textarea", {
			textContent: text,
			style: { width: "70vw", height: "70vh" },
		});

		const input = $el("input", {
			style: { flex: "auto" },
			value: DEFAULT_CUSTOM_WORDS_URL,
		});

		super.show(
			$el("div", { style: { display: "flex", flexDirection: "column", overflow: "hidden", maxHeight: "100%" } }, [
				$el("h2", { textContent: "Custom Autocomplete Words", style: { color: "#fff", marginTop: 0, textAlign: "center", fontFamily: "sans-serif" } }),
				$el("div", { style: { color: "#fff", fontFamily: "sans-serif", display: "flex", alignItems: "center", gap: "5px" } }, [
					$el("label", { textContent: "Load Custom List: " }),
					input,
					$el("button", {
						textContent: "Load",
						onclick: async () => {
							try {
								const res = await fetch(input.value);
								if (res.status !== 200) throw new Error(`Error loading: ${res.status} ${res.statusText}`);
								this.words.value = await res.text();
							} catch (_e) {
								alert("Error loading custom list, try manually copy + pasting the list");
							}
						},
					}),
				]),
				this.words,
			])
		);
	}

	createButtons() {
		const btns = super.createButtons();
		const save = $el("button", {
			type: "button",
			textContent: "Save",
			onclick: async () => {
				try {
					const res = await api.fetchApi("/booruhelper/autocomplete", { method: "POST", body: this.words.value });
					if (res.status !== 200) throw new Error(`Error saving: ${res.status} ${res.statusText}`);
					save.textContent = "Saved!";
					addCustomWords(this.words.value);
					setTimeout(() => (save.textContent = "Save"), 500);
				} catch (_e) {
					alert("Error saving word list!");
				}
			},
		});
		btns.unshift(save);
		return btns;
	}
}

app.registerExtension({
	name: id,
	init() {
		const STRING = ComfyWidgets.STRING;
		const SKIP_WIDGETS = new Set(["ttN xyPlot.x_values", "ttN xyPlot.y_values"]);

		ComfyWidgets.STRING = function (node, inputName, inputData) {
			const r = STRING.apply(this, arguments);
			if (inputData[1]?.multiline) {
				const config = inputData[1]?.["pysssss.autocomplete"];
				if (config === false) return r;

				const widgetId = `${node.comfyClass}.${inputName}`;
				if (SKIP_WIDGETS.has(widgetId)) return r;

				let words;
				let separator;
				if (typeof config === "object") {
					separator = config.separator;
					words = {};
					if (config.words) {
						Object.assign(words, TextAreaAutoComplete.groups[node.comfyClass + "." + inputName] ?? {});
					}
					for (const item of config.groups ?? []) {
						if (item === "*") {
							Object.assign(words, TextAreaAutoComplete.globalWords);
						} else {
							Object.assign(words, TextAreaAutoComplete.groups[item] ?? {});
						}
					}
				}

				new TextAreaAutoComplete(r.widget.inputEl ?? r.widget.element, words, separator);
			}
			return r;
		};

		TextAreaAutoComplete.globalSeparator = localStorage.getItem(id + ".AutoSeparate") ?? ", ";
		const enabledSetting = app.ui.settings.addSetting({
			id,
			name: "Booru Helper Autocomplete",
			defaultValue: true,
			type: (name, setter, value) =>
				$el("tr", [
					$el("td", [$el("label", { for: id.replaceAll(".", "-"), textContent: name })]),
					$el("td", [
						$el("label", { textContent: "Enabled ", style: { display: "block" } }, [
							$el("input", {
								id: id.replaceAll(".", "-"),
								type: "checkbox",
								checked: value !== false,
								onchange: (event) => {
									const checked = !!event.target.checked;
									TextAreaAutoComplete.enabled = checked;
									setter(checked);
								},
							}),
						]),
						$el("label", { textContent: "Auto-insert comma ", style: { display: "block" } }, [
							$el("input", {
								type: "checkbox",
								checked: !!TextAreaAutoComplete.globalSeparator,
								onchange: (event) => {
									const checked = !!event.target.checked;
									TextAreaAutoComplete.globalSeparator = checked ? ", " : "";
									localStorage.setItem(id + ".AutoSeparate", TextAreaAutoComplete.globalSeparator);
								},
							}),
						]),
						$el("label", { textContent: "Replace _ with space ", style: { display: "block" } }, [
							$el("input", {
								type: "checkbox",
								checked: !!TextAreaAutoComplete.replacer,
								onchange: (event) => {
									const checked = !!event.target.checked;
									TextAreaAutoComplete.replacer = checked ? (v) => v.replaceAll("_", " ") : undefined;
									localStorage.setItem(id + ".ReplaceUnderscore", checked);
								},
							}),
						]),
						$el("label", { textContent: "Max suggestions: ", style: { display: "block" } }, [
							$el("input", {
								type: "number",
								value: +TextAreaAutoComplete.suggestionCount,
								style: { width: "80px" },
								onchange: (event) => {
									const value = +event.target.value;
									TextAreaAutoComplete.suggestionCount = value;
									localStorage.setItem(id + ".SuggestionCount", TextAreaAutoComplete.suggestionCount);
								},
							}),
						]),
						$el("button", {
							textContent: "Manage Custom Words",
							onclick: () => new CustomWordsDialog().show(),
							style: { fontSize: "14px", display: "block", marginTop: "5px" },
						}),
					]),
				]),
		});

		TextAreaAutoComplete.enabled = enabledSetting.value !== false;
		TextAreaAutoComplete.replacer = localStorage.getItem(id + ".ReplaceUnderscore") === "true" ? (v) => v.replaceAll("_", " ") : undefined;
		TextAreaAutoComplete.insertOnTab = localStorage.getItem(id + ".InsertOnTab") !== "false";
		TextAreaAutoComplete.insertOnEnter = localStorage.getItem(id + ".InsertOnEnter") !== "false";
		TextAreaAutoComplete.suggestionCount = +localStorage.getItem(id + ".SuggestionCount") || 20;
	},
	setup() {
		addCustomWords();
	},
	beforeRegisterNodeDef(_, def) {
		const inputs = { ...def.input?.required, ...def.input?.optional };
		for (const input in inputs) {
			const config = inputs[input][1]?.["pysssss.autocomplete"];
			if (!config) continue;
			if (typeof config === "object" && config.words) {
				const words = {};
				for (const text of config.words || []) {
					const obj = typeof text === "string" ? { text } : text;
					words[obj.text] = obj;
				}
				TextAreaAutoComplete.updateWords(def.name + "." + input, words, false);
			}
		}
	},
});
