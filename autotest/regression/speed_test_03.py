import copy
import os
import time
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.mf6 import (
    ExtFileAction,
    MFModel,
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfdisv,
    ModflowGwfdrn,
    ModflowGwfevt,
    ModflowGwfevta,
    ModflowGwfghb,
    ModflowGwfgnc,
    ModflowGwfgwf,
    ModflowGwfgwt,
    ModflowGwfhfb,
    ModflowGwfic,
    ModflowGwfmaw,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrch,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfsfr,
    ModflowGwfsto,
    ModflowGwfwel,
    ModflowGwtadv,
    ModflowGwtdis,
    ModflowGwtic,
    ModflowGwtmst,
    ModflowGwtoc,
    ModflowGwtssm,
    ModflowIms,
    ModflowTdis,
    ModflowUtltas,
)
from flopy.mf6.data.mfdatastorage import DataStorageType
from flopy.mf6.mfbase import FlopyException, MFDataException
from flopy.mf6.utils import testutils
from flopy.utils import CellBudgetFile
from flopy.utils.compare import compare_concentrations, compare_heads
from flopy.utils.datautil import PyListUtil

# init paths
#example_data_path = "test"
function_tmpdir = "."
test_ex_name = "np001"
model_name = "np001_mod"
#sim_path = "test"


def create_sim(sim_path, pandas_support, well_spd, drn_data, riv_spd,
               maw_package, maw_connection, maw_spd, sfr_rec,
               reach_con_rec, reach_div_rec, reach_per_rec, binary):
    # create simulation
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=sim_path,
        write_headers=False,
    )
    sim.simulation_data.lazy_io = True
    sim.simulation_data.pandas_support = pandas_support
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis_package = ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )

    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename=f"{test_ex_name}.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )

    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )

    # specifying dis package twice with the same name should automatically
    # remove the old dis package
    #top = {"filename": "top.bin", "data": 100.0, "binary": True}
    #botm = {"filename": "botm.bin", "data": 50.0, "binary": True}
    dis_package = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1000,
        ncol=1000,
        delr=500.0,
        delc=500.0,
        top=100.0,
        botm=50.0,
        filename=f"{model_name}.dis",
        pname="mydispkg",
    )
    ic_package = ModflowGwfic(
        model, strt=80.0, filename=f"{model_name}.ic"
    )

    npf_package = ModflowGwfnpf(
        model,
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=5.0,
    )

    """    
    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("budget", "fileout", "np001_mod 1.cbc",)],
        head_filerecord=[("head", "fileout", "np001_mod 1.hds",)],
        saverecord={
                0: [("SAVE", "HEAD", "ALL"), ("SAVE", "BUDGET", "ALL")],
            1: [],
        },
        printrecord=[("PRINT", "HEAD", "ALL")],
    )
    empty_sp_text = oc_package.saverecord.get_file_entry(1)
    assert empty_sp_text == ""
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("PRINT", "HEAD", "ALL"),
                                     ("PRINT", "BUDGET", "ALL")], 1)
    oc_package.saverecord.set_data([("SAVE", "HEAD", "ALL"),
                                    ("SAVE", "BUDGET", "ALL")], 1)
    """

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("np001_mod 1.cbc",)],
        head_filerecord=[("np001_mod 1.hds",)],
        saverecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            1: [],
        },
        printrecord=[("HEAD", "ALL")],
    )
    empty_sp_text = oc_package.saverecord.get_file_entry(1)
    assert empty_sp_text == ""
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)
    oc_package.saverecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)

    sto_package = ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15
    )
    """
    # test saving a test file with list data
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1000000,
        stress_period_data=well_spd,
    )

    # test getting data from a binary file
    drn_spd = {0: {"data": drn_data},
               1: {"data": drn_data}}
    drn_package = ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        #boundnames=True,
        maxbound=1000000,
        stress_period_data=drn_spd)

    riv_package = ModflowGwfriv(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1000000,
        auxiliary=["var1", "var2", "var3"],
        stress_period_data=riv_spd,
    )
    """
    connection = {"filename": "mawcon.txt", "data": maw_connection}
    maw_package = ModflowGwfmaw(
        model,
        print_input=True,
        nmawwells=len(maw_package),
        packagedata=maw_package,
        connectiondata=connection,
        perioddata=maw_spd
    )

    sfr_package = ModflowGwfsfr(
        model,
        unit_conversion=1.486,
        #stage_filerecord="test.sfr.stage.bin",
        #budget_filerecord="test.sfr.cbc",
        nreaches=len(sfr_rec),
        packagedata=sfr_rec,
        connectiondata=reach_con_rec,
        diversions=reach_div_rec,
        perioddata={0: reach_per_rec},
    )

    return sim

"""
fd_test_1 = open("con.maw", "w")
fd_test_2 = open("np001_mod.maw", "w")
fd_test_3 = open("connp.maw", "w")
fd_test_4 = open("connp.txt", "w")
fd_test_5 = open("npcon.maw", "w")
fd_test_1.write("test")
fd_test_2.write("test")
"""

binary = False

well_data = []
for row in range(0, 1000):
    for col in range(0, 1000):
        well_data.append((0, row, col, -0.1))
well_spd = {
    0: {
        "filename": "wel_0.txt",
        "binary": binary,
        "data": well_data,
    },
    1: None,
}
drn_data = []
for row in range(0, 1000):
    for col in range(0, 1000):
        #drn_data.append((0, row, col, 100.0, 80.0, f"bn_{row*1000+col}"))
        drn_data.append((0, row, col, 100.0, 80.0))

riv_data = []
for row in range(0, 1000):
    for col in range(0, 1000):
        riv_data.append((0, row, col, 110.0, 90.0, 100.0, 1.0, 2.0, 3.0))
riv_spd = {
    0: {
        "filename": os.path.join("riv_folder", "riv.txt"),
        "data": riv_data,
        "binary": binary
    }
}

maw_package = []
well_id = 0
for row in range(0, 1000):
    for col in range(0, 1000):
        maw_package.append((well_id, 10.0, 50.0, 80.0, "SKIN", 1))
        well_id += 1
pac_columns = ["wellno", "radius", "bottom", "strt", "condeqn", "ngwfnodes"]
maw_package_df = pd.DataFrame(maw_package, columns=pac_columns)
#maw_package2 = copy.deepcopy(maw_package)
maw_connection = []
well_id = 0
for row in range(0, 1000):
    for col in range(0, 1000):
        maw_connection.append((well_id, 0, 0, row, col, 100.0, 50.0, 1.0,
                               20.0))
        well_id += 1
con_columns = ["wellno", "icon", "layer", "row", "column", "scrn_top",
               "scrn_bot", "hk_skin", "radius_skin"]
maw_con_df = pd.DataFrame(maw_connection, columns=con_columns)

#maw_connection2 = copy.deepcopy(maw_connection)
maw_spd = {}
maw_spd_df = {}
status_columns = ["wellno", "status", "active"]
rate_columns = ["wellno", "rate", "amount"]
scaling_columns = ["wellno", "rate_scaling", "amt1", "amt2"]
maw_all = []
maw_status = []
maw_rate = []
maw_rs = []
for row in range(0, 1000):
    for col in range(0, 1000):
        maw_all.append((well_id, "STATUS", "ACTIVE"))
        maw_all.append((well_id, "RATE", -0.01))
        maw_all.append((well_id, "RATE_SCALING", 60.0, 5.0))
        maw_status.append((well_id, "STATUS", "ACTIVE"))
        maw_rate.append((well_id, "RATE", -0.01))
        maw_rs.append((well_id, "RATE_SCALING", 60.0, 5.0))
        well_id += 1
maw_spd[0] = maw_all
maw_spd[1] = maw_all
maw_status_df = pd.DataFrame(maw_status, columns=status_columns)
maw_rate_df = pd.DataFrame(maw_rate, columns=rate_columns)
maw_rs_df = pd.DataFrame(maw_rs, columns=scaling_columns)
maw_spd_df[0] = [maw_status_df, maw_rate_df, maw_rs_df]
maw_spd_df[1] = [copy.deepcopy(maw_status_df), copy.deepcopy(maw_rate_df),
                 copy.deepcopy(maw_rs_df)]

#maw_spd2 = copy.deepcopy(maw_spd)


pth = os.path.join("..", "..", "examples", "data", "mf6", "create_tests",
                   "test028_sfr")
sfr_rec = testutils.read_sfr_rec(os.path.join(pth, "sfr_rec.txt"), 3, False)
sfr_rec_tpl = testutils.read_sfr_rec(os.path.join(pth, "sfr_rec.txt"), 3, True)
reach_con_rec = testutils.read_reach_con_rec(
    os.path.join(pth, "sfr_reach_con_rec.txt")
)
reach_div_rec = testutils.read_reach_div_rec(
    os.path.join(pth, "sfr_reach_div_rec.txt")
)
reach_per_rec = testutils.read_reach_per_rec(
    os.path.join(pth, "sfr_reach_per_rec.txt")
)

# time with pandas support
before_pandas_write = time.perf_counter()
sim_pandas = create_sim("pandas_2", True, well_spd, drn_data, riv_spd,
                        maw_package_df, maw_con_df, maw_spd_df, sfr_rec,
                        reach_con_rec, reach_div_rec, reach_per_rec, binary)
after_pandas_sort = time.perf_counter()
sim_pandas.write_simulation()
after_pandas_write = time.perf_counter()
sim_l = MFSimulation.load(sim_ws="pandas_2")
model_l = sim_l.get_model()
#wel_l = model_l.get_package("wel")
#wel_l_spd = wel_l.stress_period_data.get_data()
#riv_l = model_l.get_package("riv")
#riv_l_spd = riv_l.stress_period_data.get_data()
#drn_l = model_l.get_package("drn")
#drn_l_spd = drn_l.stress_period_data.get_data()
maw_l = model_l.get_package("maw")
maw_l_spd = maw_l.perioddata.get_data()
sfr_l = model_l.get_package("sfr")
sfr_l_pac = sfr_l.packagedata.get_data()
after_pandas_read = time.perf_counter()
#sim_pandas.run_simulation()

# time without pandas support
before_reg_write = time.perf_counter()
sim_reg = create_sim("reg_2", False, well_spd, drn_data, riv_spd,
                     maw_package, maw_connection, maw_spd, sfr_rec_tpl,
                     reach_con_rec, reach_div_rec, reach_per_rec, binary)
sim_reg.write_simulation()
after_reg_write = time.perf_counter()
sim_l_reg = MFSimulation.load(sim_ws="reg_2", pandas_support=False)
model_l_reg = sim_l_reg.get_model()
#wel_l_reg = model_l_reg.get_package("wel")
#wel_spd_reg = wel_l_reg.stress_period_data.get_data()
#riv_l_reg = model_l_reg.get_package("riv")
#riv_spd_reg = riv_l_reg.stress_period_data.get_data()
#drn_l_reg = model_l_reg.get_package("drn")
#drn_spd_reg = drn_l_reg.stress_period_data.get_data()
maw_l_reg = model_l.get_package("maw")
maw_spd_reg = maw_l.perioddata.get_data()
sfr_l_reg = model_l.get_package("sfr")
sfr_pac_reg = sfr_l.packagedata.get_data()
after_reg_read = time.perf_counter()
#sim_reg.run_simulation()

print(f"Pandas write time: {after_pandas_write - before_pandas_write}")
print(f"Pandas sort time: {after_pandas_sort - before_pandas_write}")
print(f"Reg write time: {after_reg_write - before_reg_write}")

print(f"Pandas read time: {after_pandas_read - after_pandas_write}")
print(f"Reg read time: {after_reg_read - after_reg_write}")
