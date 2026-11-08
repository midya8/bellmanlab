import copy,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from examples.render_robust_policy import render
from bellmanlab.robust_policy_demo import robust_policy_demo
NS={'s':'http://www.w3.org/2000/svg'}


class RobustPolicyFigureTests(unittest.TestCase):
    def test_every_displayed_result_cell_matches_exact_result(self):
        d=robust_policy_demo();root=ET.fromstring(render(d));cells=[t for t in root.findall('s:text',NS) if 'data-case' in t.attrib];self.assertEqual(len(cells),15)
        for cell in cells:
            c=d[['inner_capped','controller_capped','complete'][int(cell.attrib['data-case'])]];value=c[cell.attrib['data-field']];self.assertEqual(json.loads(cell.attrib['data-exact']),value)
            if cell.attrib['data-field'] in ['policy_value_exact','optimal']:self.assertEqual(cell.text,'yes' if value else 'no')
            elif cell.attrib['data-field']=='values':self.assertEqual(cell.text,'/'.join(map(str,value[0])))
            else:self.assertEqual(cell.text,str(value))
    def test_full_exact_metadata(self):self.assertEqual(json.loads(ET.fromstring(render(robust_policy_demo())).find('s:metadata',NS).text),robust_policy_demo())
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'robust-policy-audit.svg').read_text(),render(robust_policy_demo()))
    def test_reject_changed_nested_undisplayed_certificate(self):
        d=copy.deepcopy(robust_policy_demo());d['complete']['history'][0]['evaluation']['history'][0]['residual_sup'][0]+=1
        with self.assertRaises(ValueError):render(d)
