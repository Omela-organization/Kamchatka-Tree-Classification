import schedule
from src.tree_or_not_tree import run

schedule.every(1).day.do(run)
while True:
    schedule.run_pending()
