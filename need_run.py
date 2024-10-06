import baby_rdp as rp
import pandas as pd
import Baby_washdata_change as wdc

def main():
    data_init=wdc.trans_data("washed_data.csv")
    data = data_init.apply(lambda row: rp.simplify(row, 0.013), axis=1)
    data.to_csv("simple_path.csv")
    return

if __name__ == "__main__":
    main()