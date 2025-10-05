// app/static/js/chart_helpers.js
(function () {
  function renderDatasetsTable(containerId, datasets, opts = {}) {
    const xLabel = opts.xLabel || 'X';
    const valueFmt = opts.valueFmt || (v => (v == null ? '' : String(v)));
    const xFmt = opts.xFmt || (v => v);
    const typeGlyph = t => (t === 'forecast' ? '▲' : (t === 'extrapolation' ? '■' : '●'));

    const xValuesSet = new Set();
    (datasets || []).forEach(ds => (ds.data || []).forEach(p => xValuesSet.add(p.x)));
    const xValues = Array.from(xValuesSet).sort((a, b) => a - b);

    const byLabel = {};
    (datasets || []).forEach(ds => {
      const m = {};
      (ds.data || []).forEach(p => { m[p.x] = p; });
      byLabel[ds.label] = m;
    });

    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';

    const tbl = document.createElement('table');
    tbl.className = 'series-table';

    const thead = document.createElement('thead');
    const hrow = document.createElement('tr');
    const thX = document.createElement('th');
    thX.textContent = xLabel;
    hrow.appendChild(thX);
    (datasets || []).forEach(ds => {
      const th = document.createElement('th');
      th.textContent = ds.label;
      hrow.appendChild(th);
    });
    thead.appendChild(hrow);
    tbl.appendChild(thead);

    const tbody = document.createElement('tbody');
    xValues.forEach(x => {
      const tr = document.createElement('tr');
      const tdX = document.createElement('td');
      tdX.textContent = xFmt(x);
      tr.appendChild(tdX);

      (datasets || []).forEach(ds => {
        const td = document.createElement('td');
        const p = (byLabel[ds.label] || {})[x];
        if (p) {
          td.textContent = valueFmt(p.y);
          const s = document.createElement('span');
          s.textContent = ' ' + typeGlyph(p.t || p.type || '');
          s.title = p.t || p.type || '';
          td.appendChild(s);
        } else {
          td.textContent = '';
        }
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    el.appendChild(tbl);
  }

  // export
  window.renderDatasetsTable = renderDatasetsTable;
})();

// --- Type legend (history/forecast/extrapolation) ---
(function () {
  function renderTypeLegend(containerId, labels) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const L = Object.assign({
      history: 'Raunmæling',
      forecast: 'Spá',
      extrapolation: 'Framreikningur'
    }, labels || {});
    el.innerHTML = `
      <div class="legend">
        <div class="legend-item"><span class="glyph history">●</span> ${L.history}</div>
        <div class="legend-item"><span class="glyph forecast">▲</span> ${L.forecast}</div>
        <div class="legend-item"><span class="glyph extrapolation">■</span> ${L.extrapolation}</div>
      </div>
    `;
  }

  window.renderTypeLegend = renderTypeLegend;
})();

