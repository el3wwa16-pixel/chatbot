"""Accuracy check from the command line:  python evaluate.py"""
import config
from bot import evaluate
from bot.faq import FAQ
from bot.rag import KnowledgeBase

report = evaluate.run(KnowledgeBase(config.KNOWLEDGE_PATH), FAQ(config.FAQ_PATH), config.EVAL_PATH)
print(f"Accuracy: {report['accuracy']}%  ({report['passed']} of {report['total']})")
for kind, numbers in report["by_type"].items():
    print(f"  {kind:6s} {numbers['passed']} / {numbers['total']}")
for row in report["failures"]:
    print(f"  FAILED [{row['type']}] {row['question']}  expected: {row['expected']}  got: {row['got']}")
