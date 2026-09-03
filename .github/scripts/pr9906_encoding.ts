// Encoding integrity for the strings unslothai/unsloth#9906 adds to all 12 locale
// catalogs. Pinned by codepoint, not by eyeballing: a CRLF rewrite, a lost BOM
// decision or a mojibaked checkout on Windows or macOS changes these values, and
// every other frontend gate would still pass.

import assert from "node:assert/strict";
import test from "node:test";

const ROOT = "../../studio/frontend/src/i18n/locales/";
const EXPECTED: Record<string, string> = {
  en: "Couldn't copy automatically. Select the token and copy manually.",
  ar: "تعذّر النسخ تلقائيًا. حدّد التوكن وانسخه يدويًا.",
  de: "Automatisches Kopieren fehlgeschlagen. Token markieren und manuell kopieren.",
  es: "No se pudo copiar automáticamente. Selecciona el token y cópialo manualmente.",
  fr: "Copie automatique impossible. Sélectionnez le jeton et copiez-le manuellement.",
  hi: "स्वतः कॉपी नहीं हो सका। टोकन चुनें और मैन्युअल रूप से कॉपी करें।",
  it: "Copia automatica non riuscita. Seleziona il token e copialo manualmente.",
  ja: "自動でコピーできませんでした。トークンを選択して手動でコピーしてください。",
  ko: "자동으로 복사하지 못했습니다. 토큰을 선택해 직접 복사하세요.",
  "pt-br": "Não foi possível copiar automaticamente. Selecione o token e copie manualmente.",
  ru: "Не удалось скопировать автоматически. Выделите токен и скопируйте вручную.",
  "zh-CN": "无法自动复制。请选中 token 并手动复制。",
};
const EXPORT: Record<string, string> = { "pt-br": "ptBR", "zh-CN": "zhCN" };

test("every locale's copy-failure string survives this platform's checkout", async () => {
  for (const [name, expected] of Object.entries(EXPECTED)) {
    const mod: any = await import(new URL(`${ROOT}${name}.ts`, import.meta.url).href);
    const actual = mod[EXPORT[name] ?? name]?.settings?.apiKeys?.copyAccessTokenFailed;
    assert.equal(actual, expected, `${name}: string differs on this platform`);
    assert.doesNotMatch(actual, /\r/, `${name}: carriage return in the string`);
    assert.ok(!actual.includes("�"), `${name}: replacement char, checkout mojibaked`);
    console.log(`  ${name.padEnd(6)} ok (${[...actual].length} codepoints)`);
  }
});

test("the source files themselves are LF and BOM-free on this platform", async () => {
  const { readFileSync } = await import("node:fs");
  for (const name of Object.keys(EXPECTED)) {
    const p = new URL(`${ROOT}${name}.ts`, import.meta.url);
    const buf = readFileSync(p);
    assert.ok(!(buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf), `${name}: BOM present`);
    assert.equal(buf.includes(0x0d), false, `${name}: CRLF in a frontend source`);
  }
});
