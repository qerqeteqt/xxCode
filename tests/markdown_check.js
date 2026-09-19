/* Markdown 渲染器的断言脚本。由 tests/test_markdown.py 用 node 跑。
 *
 * 为什么用 node 而不是在浏览器里点点看：**这是唯一一处我们自己拼 HTML 的地方**，
 * 而拼错的后果是 XSS（模型输出是不可信内容）。手工点几下测不出 `<script>` 有没有被转义。
 */

const { renderMarkdown, inline, esc } = require("../app/web/static/markdown.js");

let failed = 0;

function check(name, actual, expected) {
  const ok = expected instanceof RegExp ? expected.test(actual) : actual === expected;
  if (ok) return;
  failed++;
  console.log(`✗ ${name}`);
  console.log(`    期望: ${expected}`);
  console.log(`    实际: ${actual}`);
}

function contains(name, actual, needle) {
  if (actual.includes(needle)) return;
  failed++;
  console.log(`✗ ${name}\n    实际里没有: ${needle}\n    实际: ${actual}`);
}

function lacks(name, actual, needle) {
  if (!actual.includes(needle)) return;
  failed++;
  console.log(`✗ ${name}\n    实际里不该有: ${needle}\n    实际: ${actual}`);
}

// ---------------------------------------------------------------- 安全

// 这一组是整个渲染器存在的意义：模型输出是不可信内容
check("HTML 被转义",
  renderMarkdown("<script>alert(1)</script>"),
  /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);

lacks("行内的 <img onerror> 不该变成标签",
  renderMarkdown('<img src=x onerror=alert(1)>'), "<img");

check("代码块里的 HTML 也转义",
  renderMarkdown("```html\n<b>粗</b>\n```"),
  /&lt;b&gt;粗&lt;\/b&gt;/);

lacks("javascript: 链接不生成 a 标签",
  renderMarkdown("[点我](javascript:alert(1))"), "<a ");

contains("http 链接正常生成",
  renderMarkdown("[文档](https://example.com/x)"),
  '<a href="https://example.com/x" target="_blank" rel="noopener noreferrer">文档</a>');

check("esc 覆盖五种字符",
  esc(`&<>"'`), "&amp;&lt;&gt;&quot;&#39;");

// ---------------------------------------------------------------- 块级

contains("h1", renderMarkdown("# 大标题"), "<h1>大标题</h1>");
contains("h2", renderMarkdown("## 写法一"), "<h2>写法一</h2>");
contains("h3", renderMarkdown("### 小标题"), "<h3>小标题</h3>");
contains("h6 封顶", renderMarkdown("###### 很深"), "<h6>很深</h6>");

contains("围栏代码块", renderMarkdown("```python\nx = 1\n```"), "<pre><code>x = 1</code>");
contains("代码块带语言标签", renderMarkdown("```python\nx = 1\n```"),
  '<span class="lang">python</span>');
contains("代码块有复制按钮", renderMarkdown("```\nx\n```"), "data-copy");
lacks("没写语言就不显示标签", renderMarkdown("```\nx\n```"), 'class="lang"');

contains("无序列表", renderMarkdown("- 一\n- 二"), "<ul><li>一</li><li>二</li></ul>");
contains("有序列表", renderMarkdown("1. 一\n2. 二"), "<ol><li>一</li><li>二</li></ol>");
contains("分隔线", renderMarkdown("---"), "<hr>");
contains("引用", renderMarkdown("> 引用的话"), "<blockquote>");
contains("段落", renderMarkdown("普通一行"), "<p>普通一行</p>");
contains("段落内换行变 br", renderMarkdown("第一行\n第二行"), "<p>第一行<br>第二行</p>");
contains("空行分段", renderMarkdown("甲\n\n乙"), "<p>甲</p><p>乙</p>");

// ---------------------------------------------------------------- 行内

contains("加粗", renderMarkdown("**重点**"), "<strong>重点</strong>");
contains("斜体", renderMarkdown("这是 *强调* 的部分"), "<em>强调</em>");
contains("行内代码", renderMarkdown("用 `useState` 就行"), "<code>useState</code>");

// 行内代码必须先摘出来，否则里面的 ** 会被加粗规则吃掉
lacks("行内代码里的星号不该变加粗",
  renderMarkdown("`a ** b`"), "<strong>");
contains("行内代码里的星号原样保留",
  renderMarkdown("`a ** b`"), "a ** b");

// ---------------------------------------------------------------- 混排

const mixed = renderMarkdown([
  "N 皇后是个经典问题。",
  "",
  "## 写法一：最直观",
  "",
  "思路是**逐格试探**，用 `board[i]` 存列号：",
  "",
  "```python",
  "def solve(n):",
  '    return []  # <-- 这里有个 <尖括号>',
  "```",
  "",
  "1. 检查列冲突",
  "2. 检查对角线",
  "",
  "> 注意：三个写法差别在剪枝。",
  "",
  "---",
].join("\n"));

contains("混排：段落", mixed, "<p>N 皇后是个经典问题。</p>");
contains("混排：标题", mixed, "<h2>写法一：最直观</h2>");
contains("混排：行内代码", mixed, "<code>board[i]</code>");
contains("混排：代码块内容", mixed, "def solve(n):");
contains("混排：代码块里的尖括号被转义", mixed, "&lt;尖括号&gt;");
lacks("混排：不该冒出真的尖括号标签", mixed, "<尖括号>");
contains("混排：有序列表", mixed, "<ol>");
contains("混排：引用", mixed, "<blockquote>");
contains("混排：分隔线", mixed, "<hr>");

// ---------------------------------------------------------------- 流式

// 流式时每一小段都会重渲染一次，未闭合的围栏必须能显示而不是崩掉
contains("未闭合的代码围栏也能渲染", renderMarkdown("```python\nx = 1"), "<pre>");

console.log(failed === 0 ? "OK" : `FAILED: ${failed}`);
process.exit(failed === 0 ? 0 : 1);
