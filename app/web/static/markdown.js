/* 极简 Markdown 渲染器。

   为什么自己写而不是引库（marked / DOMPurify）：
     1. 要能离线跑 —— 从 CDN 加载意味着断网时整个页面渲染不出来
     2. 安全上更可控 —— **先把所有 HTML 转义掉，再只插入我们自己生成的标签**。
        模型输出里写 <script> 只会显示成字面量，不可能变成标签。
        引库的话得额外配一个 sanitizer，配错了就是 XSS —— 而这是本地跑、
        能读写你文件的 agent，XSS 的后果比一般网页严重得多
   代价是支持的语法有限，但覆盖模型实际会写的那些足够了。

   抽成独立文件而不是内联在 index.html 里，是为了**能测** ——
   tests/test_markdown.py 会用 node 跑它。
*/

"use strict";

/* 为什么自己写而不是引库（marked / DOMPurify）：
     1. 要能离线跑 —— 从 CDN 加载意味着断网时整个页面渲染不出来
     2. 安全上更可控 —— **先把所有 HTML 转义掉，再只插入我们自己生成的标签**。
        模型输出里写 <script> 只会显示成字面量，不可能变成标签。
        引库的话得额外配一个 sanitizer，配错了就是 XSS，而这是本地跑、
        能读写你文件的 agent，XSS 的后果比一般网页严重
   代价是支持的语法有限，但覆盖模型实际会写的那些足够了。 */

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** 行内规则。**入参必须是已转义的文本** —— 这样我们插入的标签就是唯一的标签。 */
function inline(escaped) {
  // 行内代码先摘出来，否则里面的 ** 会被后面的规则吃掉
  const codes = [];
  let s = escaped.replace(/`([^`\n]+)`/g, (_, c) => {
    codes.push(c);
    return "\u0000" + (codes.length - 1) + "\u0000";
  });

  s = s
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    // 只认 http(s)：javascript: 那种链接一律不生成
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

const inlineRaw = (raw) => inline(esc(raw));

/** 行内识别器：这些开头的行不该被并进段落。 */
const BLOCK_START = /^\s*(```|#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\s*([-*_])\2{2,}\s*$)/;

function renderMarkdown(src) {
  // 按原始行解析、在**输出的时候**才转义 —— 先转义的话，
  // `>` 会变成 `&gt;`，引用块的识别就失效了
  const lines = String(src).split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // 围栏代码块
    const fence = line.match(/^\s*```(\S*)\s*$/);
    if (fence) {
      const lang = fence[1];
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++; // 吃掉收尾的 ```
      out.push(
        `<pre><code>${esc(body.join("\n"))}</code>` +
        (lang ? `<span class="lang">${esc(lang)}</span>` : "") +
        `<button class="copy" data-copy>复制</button></pre>`
      );
      continue;
    }

    // 标题：直接映射（# -> h1）。不做「降一级」那种偏移 ——
    // 内容区本来就没有 h1，偏移只会让 ## 变得比预期小一号
    const heading = line.match(/^\s*(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 6);
      out.push(`<h${level}>${inlineRaw(heading[2])}</h${level}>`);
      i++;
      continue;
    }

    // 分隔线
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { out.push("<hr>"); i++; continue; }

    // 引用
    if (/^\s*>/.test(line)) {
      const body = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        body.push(lines[i++].replace(/^\s*>\s?/, ""));
      }
      out.push(`<blockquote>${renderMarkdown(body.join("\n"))}</blockquote>`);
      continue;
    }

    // 列表
    const ordered = /^\s*\d+[.)]\s+/.test(line);
    if (ordered || /^\s*[-*+]\s+/.test(line)) {
      const re = ordered ? /^\s*\d+[.)]\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(re);
        if (!m) break;
        items.push(`<li>${inlineRaw(m[1])}</li>`);
        i++;
      }
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
      continue;
    }

    if (!line.trim()) { i++; continue; }

    // 段落：连续的非块级行并成一段，段内换行用 <br>
    const para = [];
    while (i < lines.length && lines[i].trim() && !BLOCK_START.test(lines[i])) {
      para.push(lines[i++]);
    }
    out.push(`<p>${para.map(inlineRaw).join("<br>")}</p>`);
  }
  return out.join("");
}

/* 浏览器里挂到 window，node 里走 module.exports —— 同一个文件两边都能用 */
if (typeof window !== "undefined") window.renderMarkdown = renderMarkdown;
if (typeof module !== "undefined" && module.exports) {
  module.exports = { renderMarkdown, esc, inline };
}
