import os
import shutil
import subprocess
import textwrap

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_transform_live_accepts_pipeline_briefing_bullets():
    node = shutil.which("node")
    if not node:
        pytest.skip("node unavailable")
    script = textwrap.dedent(r"""
        const fs = require('fs');
        const html = fs.readFileSync('index.html', 'utf8');
        const m = html.match(/<script type="text\/x-dc"[\s\S]*?>([\s\S]*?)<\/script>/);
        if (!m) throw new Error('component script not found');
        globalThis.DCLogic = class {};
        eval(m[1] + '\nglobalThis.__Component = Component;');
        const component = new globalThis.__Component();
        component.props = {};
        const doc = JSON.parse(fs.readFileSync('mena_data.json', 'utf8'));
        const out = component.transformLive(doc);
        if (out.meta.mainIndex !== doc.meta.main_index) throw new Error('live meta not used');
        if (!out.brief.bullets.every(b => typeof b.text === 'string' && typeof b.cat === 'string')) {
            throw new Error('brief bullets were not normalized');
        }
    """)
    subprocess.run([node, "-e", script], cwd=ROOT, check=True)
