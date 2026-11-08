import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from examples.render_audit import render,frozen
from bellmanlab.demo import demo
NS={'s':'http://www.w3.org/2000/svg'}


class FigureTests(unittest.TestCase):
    def test_linear_scale_all_bars_and_tolerance(self):
        root=ET.fromstring(render(demo()));bars=[r for r in root.findall('s:rect',NS) if 'data-value' in r.attrib];self.assertEqual(len(bars),3)
        for bar,v in zip(bars,[.01,1,1]):
            self.assertAlmostEqual(float(bar.attrib['height']),260*v);self.assertAlmostEqual(float(bar.attrib['y']),450-260*v)
        self.assertTrue(any(p.attrib.get('data-tolerance')=='1/10' and p.attrib['d']=='M 80 424 H 1070' for p in root.findall('s:path',NS)))
    def test_full_selected_exact_certificates(self):
        d=demo();r=ET.fromstring(render(d));self.assertEqual(json.loads(r.find('s:metadata',NS).text),frozen(d));self.assertEqual([r['sweep'] for r in frozen(d)['selected_history']],[0,1,10,100,229,230])
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'bellman-audit.svg').read_text(),render(demo()))
    def test_reject_changed_history(self):
        d=copy.deepcopy(demo());d['residual_case']['iteration']['history'][3]['values'][0][0]+=1
        with self.assertRaises(ValueError):render(d)
