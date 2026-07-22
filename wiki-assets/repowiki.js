(() => {
  'use strict';
  const pagePrefix = '#/';
  let controller = null;

  function slugify(value) {
    return String(value).trim().toLowerCase()
      .replace(/<[^>]*>/g, '')
      .replace(/[\s_]+/g, '-')
      .replace(/[^\p{Letter}\p{Number}-]/gu, '')
      .replace(/-+/g, '-').replace(/^-|-$/g, '') || 'section';
  }

  function normalizeSearch(value) {
    return String(value).normalize('NFKD')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase()
      .replace(/[^\p{Letter}\p{Number}]+/gu, ' ').trim();
  }

  window.repoWiki = {
    slugify,
    normalizeSearch,
    matchesSearch(value, query) {
      const haystack = normalizeSearch(value);
      return normalizeSearch(query).split(/\s+/).filter(Boolean).every(token => haystack.includes(token));
    },
    route(path, anchor = '') {
      return pagePrefix + encodeURIComponent(path) + (anchor ? '?anchor=' + encodeURIComponent(anchor) : '');
    },
    parseRoute() {
      const raw = location.hash;
      if (!raw) return null;
      try {
        if (!raw.startsWith(pagePrefix)) return { path: decodeURIComponent(raw.replace(/^#\/?/, '')), anchor: '' };
        const [path, query = ''] = raw.slice(pagePrefix.length).split('?');
        return { path: decodeURIComponent(path), anchor: new URLSearchParams(query).get('anchor') || '' };
      } catch (_) { return null; }
    },
    encodedContentUrl(base, path) {
      return base + path.split('/').filter(Boolean).map(encodeURIComponent).join('/');
    },
    abortPrevious() {
      if (controller) controller.abort();
      controller = new AbortController();
      return controller.signal;
    },
    sanitize(markdown) {
      const template = document.createElement('template');
      template.innerHTML = marked.parse(markdown.replace(/<cite>[\s\S]*?<\/cite>/gi, ''));
      const allowed = new Set(['A','P','DIV','SPAN','H1','H2','H3','H4','H5','H6','UL','OL','LI','PRE','CODE','BLOCKQUOTE','STRONG','EM','DEL','HR','BR','TABLE','THEAD','TBODY','TR','TH','TD','IMG']);
      [...template.content.querySelectorAll('*')].forEach(el => {
        if (!allowed.has(el.tagName)) { el.replaceWith(...el.childNodes); return; }
        [...el.attributes].forEach(attr => {
          const name = attr.name.toLowerCase();
          if (name.startsWith('on') || !['href','src','alt','title','class','id'].includes(name)) el.removeAttribute(attr.name);
        });
        for (const name of ['href', 'src']) {
          const value = el.getAttribute(name);
          if (!value) continue;
          if (el.tagName === 'A' && name === 'href' && /^file:\/\//i.test(value)) {
            el.removeAttribute('href');
            el.classList.add('wiki-source-ref');
            el.title = 'Local source reference: ' + value.replace(/^file:\/\//i, '');
          } else if (!/^(?:https?:|mailto:|#|\.?\.?\/)/i.test(value)) {
            el.removeAttribute(name);
          }
        }
        if (el.tagName === 'A' && /^https?:/i.test(el.getAttribute('href') || '')) {
          el.target = '_blank'; el.rel = 'noopener noreferrer';
        }
      });
      const used = new Map();
      template.content.querySelectorAll('h1,h2,h3,h4,h5,h6').forEach(heading => {
        const base = slugify(heading.textContent);
        const count = used.get(base) || 0;
        used.set(base, count + 1);
        heading.id = count ? `${base}-${count}` : base;
      });
      return template.content;
    },
    resolveAnchor(body, value) {
      const decoded = decodeURIComponent(String(value || '')).replace(/^#/, '');
      return body.querySelector(`#${CSS.escape(decoded)}`) || body.querySelector(`#${CSS.escape(slugify(decoded))}`);
    },
    wire(body, currentPath, loadPage) {
      body.querySelectorAll('a[href]').forEach(link => link.addEventListener('click', event => {
        const href = link.getAttribute('href');
        if (href.startsWith('#')) {
          event.preventDefault();
          const requested = decodeURIComponent(href.slice(1));
          const target = this.resolveAnchor(body, requested);
          const anchor = target ? target.id : slugify(requested);
          history.replaceState(null, '', this.route(currentPath, anchor));
          target?.scrollIntoView();
        } else if (!/^[a-z]+:/i.test(href) && href.split(/[?#]/)[0].endsWith('.md')) {
          event.preventDefault();
          const target = new URL(href, 'https://wiki.invalid/' + currentPath).pathname.slice(1);
          loadPage(decodeURIComponent(target));
        }
      }));
      body.querySelectorAll('table').forEach(table => {
        if (table.parentElement?.classList.contains('wiki-table-scroll')) return;
        const wrapper = document.createElement('div');
        wrapper.className = 'wiki-table-scroll'; wrapper.tabIndex = 0;
        wrapper.setAttribute('role', 'region'); wrapper.setAttribute('aria-label', 'Scrollable table');
        table.before(wrapper); wrapper.appendChild(table);
      });
      body.querySelectorAll('pre code.language-mermaid').forEach(code => {
        const div = document.createElement('div'); div.className = 'mermaid'; div.textContent = code.textContent;
        code.parentElement.replaceWith(div);
      });
      if (window.mermaid) mermaid.run({ nodes: body.querySelectorAll('.mermaid'), suppressErrors: true });
    }
  };
})();
