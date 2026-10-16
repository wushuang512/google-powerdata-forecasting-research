"""Read aggregate research results without datasets, weights or third-party packages."""
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/name).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def main():
    google=rows("本轮方案全部指标对照.csv")
    chosen=[r for r in google if r["role"]=="DEV_frozen_R48_candidate"]
    assert len(chosen)==2
    expected={1:2.329603065559927,7:2.7011842083735202}
    print("Google R48 user-adopted research configuration (not a new blind test)")
    for r in chosen:
        h=int(r["horizon_days"]);assert abs(float(r["wape"])-expected[h])<1e-12
        print(json.dumps({"horizon_days":h,"curve_WAPE_percent":float(r["wape"]),"daily_WAPE_percent":float(r["daily_wape"]),"peak_MAPE_percent":float(r["peak_mape"])},ensure_ascii=False))
    nlr=rows("NLR严格重跑_新协议主要模型测试全指标.csv")
    lead=rows("NLR严格重跑_新协议逐预测日测试指标.csv")
    assert len(nlr)==42 and len(lead)==210
    print("NLR: 42 aggregate rows, 210 lead-day rows. See report for validation-selected versus descriptive test minima.")
    print("Dataset and model weight files are intentionally absent.")

if __name__=="__main__":main()
