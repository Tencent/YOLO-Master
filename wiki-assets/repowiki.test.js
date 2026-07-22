'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(__dirname, 'repowiki.js'), 'utf8');
const context = {
  window: {},
  location: { hash: '' },
  AbortController,
  URLSearchParams,
  decodeURIComponent,
  encodeURIComponent,
  console,
};
vm.createContext(context);
vm.runInContext(source, context);
const wiki = context.window.repoWiki;

assert.equal(wiki.slugify('Trainer API'), 'trainer-api');
assert.equal(wiki.slugify('Project Structure'), 'project-structure');
assert.equal(wiki.matchesSearch('Trainer-API Reference', 'Trainer API'), true);
assert.equal(wiki.matchesSearch('API Reference / Trainer', 'trainer api'), true);
assert.equal(wiki.matchesSearch('Predictor API', 'Trainer API'), false);

context.location.hash = wiki.route('en/API Reference.md', 'trainer-api');
assert.deepEqual(JSON.parse(JSON.stringify(wiki.parseRoute())), {
  path: 'en/API Reference.md',
  anchor: 'trainer-api',
});
assert.equal(wiki.encodedContentUrl('./wiki-content/', 'en/API Reference.md'), './wiki-content/en/API%20Reference.md');

const html = fs.readFileSync(path.join(root, 'wiki-en.html'), 'utf8');
assert.match(html, /repoWiki\.parseRoute\(\)/);
assert.match(html, /repoWiki\.matchesSearch/);
assert.match(html, /wiki-table-scroll/);
assert.match(html, /hashchange/);
assert.doesNotMatch(source, /\^file:[^\n]*test\(value\)[^\n]*removeAttribute/);

console.log('RepoWiki renderer regression checks passed.');
