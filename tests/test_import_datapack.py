import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('import_datapack',ROOT/'scripts/import_datapack.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class DatapackTests(unittest.TestCase):
    def example(self):return json.loads((ROOT/'frontend/datapacks/synthetic-demo.json').read_text())
    def test_accepts_synthetic_data(self):module.validate_datapack(self.example())
    def test_rejects_invalid_prediction(self):
        for score in [-1,2,float('nan'),True,'high']:
            data=self.example();data['chats']['example-chat']['content'][0]['prediction']=score
            with self.subTest(score=score),self.assertRaises(ValueError):module.validate_datapack(data)
    def test_cli_copies_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            command=[sys.executable,str(ROOT/'scripts/import_datapack.py'),str(ROOT/'frontend/datapacks/synthetic-demo.json'),'--output-dir',folder]
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            output=Path(folder)/'local-predictions.json'
            self.assertEqual(json.loads(output.read_text()),self.example())
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(subprocess.run(command+['--force'],capture_output=True).returncode,0)
    def test_rejects_path_traversal_and_sample_overwrite(self):
        for name in ['../outside.json','synthetic-demo.json']:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/import_datapack.py'),str(ROOT/'frontend/datapacks/synthetic-demo.json'),'--name',name],capture_output=True)
            self.assertNotEqual(result.returncode,0)
if __name__=='__main__':unittest.main()
