import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('font_subsets', ROOT / 'scripts/build_font_subsets.py')
fonts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fonts)


class FontSubsetsTest(unittest.TestCase):
    def test_rendered_regions_and_dynamic_labels_are_collected_without_script_text(self):
        parser = fonts.TextCollector()
        parser.feed('<body><header data-font-priority="critical"><h1>OxyGen Ω</h1><input aria-label="Shared clock"/><script>SECRET</script><style>INTERNAL</style></header><p>Later Ж</p></body>')
        critical = ''.join(parser.critical)
        self.assertIn('Ω', critical)
        self.assertIn('Shared clock', critical)
        self.assertNotIn('SECRET', critical)
        self.assertNotIn('INTERNAL', critical)
        self.assertNotIn('Ж', critical)
        self.assertIn('Ж', ''.join(parser.page))

    def test_generated_fonts_cover_supported_page_text_and_tiers_do_not_overlap(self):
        manifest = json.loads((fonts.OUTPUT / 'manifest.json').read_text())
        critical, page = fonts.collect(ROOT / 'dist')
        for _, family, style, _, _, sources in fonts.sources():
            expected = set().union(*(fonts.cmap(source) for source in sources)) & page
            first, rest = set(), set()
            for item in manifest['fonts']:
                if (item['family'],item['style']) != (family,style):
                    continue
                target = first if item['tier']=='critical' else rest
                target.update(fonts.cmap(fonts.OUTPUT / item['file']))
            self.assertFalse(first & rest, (family,style))
            self.assertFalse(expected-(first|rest), (family,style))
        self.assertTrue(critical)

    def test_new_rendered_glyph_is_detected_and_stale_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            html = (ROOT / 'dist/index.html').read_text().replace('OxyGen</h1>','OxyGen Ω</h1>')
            (folder/'index.html').write_text(html)
            critical,_ = fonts.collect(folder)
            self.assertIn(ord('Ω'),critical)
            check = subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/build_font_subsets.py','--check-rendered',temporary],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(check.returncode,0)
            self.assertIn('not covered',check.stderr)


if __name__ == '__main__':
    unittest.main()
