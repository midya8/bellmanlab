import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from fractions import Fraction as F
from examples.render_robust import render
from bellmanlab.robust_demo import robust_demo
NS={'s':'http://www.w3.org/2000/svg'}


class RobustFigureTests(unittest.TestCase):
    def test_all_signed_bars_use_panel_linear_scale_and_common_zero(self):
        root=ET.fromstring(render(robust_demo()));bars=[r for r in root.findall('s:rect',NS) if 'data-value' in r.attrib];self.assertEqual(len(bars),5)
        for r,v,p in zip(bars,[F(4,5),F(-4,5),F(2,5),F(-2,3),F(-2)],[0,0,0,1,1]):
            self.assertEqual(F(r.attrib['data-value']),v);self.assertEqual(int(r.attrib['data-panel']),p);scale=145 if p==0 else 72.5
            self.assertAlmostEqual(float(r.attrib['height']),float(abs(v))*scale);self.assertAlmostEqual(float(r.attrib['y']),290-max(float(v),0)*scale)
        for panel in ['0','1']:self.assertEqual(len([r for r in root.findall('s:path',NS) if r.attrib['data-panel']==panel and r.attrib['data-tick']=='0']),1)
    def test_complete_exact_demo_metadata(self):self.assertEqual(json.loads(ET.fromstring(render(robust_demo())).find('s:metadata',NS).text),robust_demo())
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'robust-audit.svg').read_text(),render(robust_demo()))
    def test_reject_undisplayed_history_change(self):
        d=copy.deepcopy(robust_demo());d['decision_case']['robust_risky_iteration']['history'][3]['scenario_q_values'][0][0][0][0]+=1
        with self.assertRaises(ValueError):render(d)
