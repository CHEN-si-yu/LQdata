"""直接运行模型脚本必须进入最终实现，不能执行复制遗留的中间入口。"""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ModelEntryPointTests(unittest.TestCase):
    def test_research_model_has_one_entry_after_effective_definitions(self):
        for folder in sorted((ROOT/'experiments').iterdir()):
            if not folder.name.startswith('V') or not folder.name[1:].isdigit() or int(folder.name[1:])<253:
                continue
            with self.subTest(model=folder.name):
                tree=ast.parse((folder/'model.py').read_text())
                guards=[n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__'
                    and any(isinstance(c,ast.Constant) and c.value=='__main__' for c in n.test.comparators)]
                self.assertEqual(len(guards),1)
                definitions=[n.lineno for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
                self.assertGreater(guards[0].lineno,max(definitions))


if __name__=='__main__':unittest.main()
