import numpy as np
import pandas as pd
import cftime
import xarray as xr

def mindist(point, array):
    '''Formula to find the minimum distance of an element in an array.'''
    mindist = np.abs(array - point).argmin()
    return mindist

def time_set_mid(ds, time_name):
    """
    set ds[time_name] to midpoint of ds[time_name].attrs['bounds'], if bounds attribute exists
    type of ds[time_name] is not changed
    ds is returned
    """

    if "bounds" not in ds[time_name].attrs:
        return ds

    # determine units and calendar of unencoded time values
    if ds[time_name].dtype == np.dtype("O"):
        units = "days since 0000-01-01"
        calendar = "noleap"
    else:
        units = ds[time_name].attrs["units"]
        calendar = ds[time_name].attrs["calendar"]

    # construct unencoded midpoint values, assumes bounds dim is 2nd
    tb_name = ds[time_name].attrs["bounds"]
    if ds[tb_name].dtype == np.dtype("O"):
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
    else:
        tb_vals = ds[tb_name].values
    tb_mid = tb_vals.mean(axis=1)

    # set ds[time_name] to tb_mid
    if ds[time_name].dtype == np.dtype("O"):
        ds[time_name] = cftime.num2date(tb_mid, units=units, calendar=calendar)
    else:
        ds[time_name] = tb_mid

    return ds

def define_pftlist():
    pftlist = [
        "not_vegetated",
        "needleleaf_evergreen_temperate_tree",
        "needleleaf_evergreen_boreal_tree",
        "needleleaf_deciduous_boreal_tree",
        "broadleaf_evergreen_tropical_tree",
        "broadleaf_evergreen_temperate_tree",
        "broadleaf_deciduous_tropical_tree",
        "broadleaf_deciduous_temperate_tree",
        "broadleaf_deciduous_boreal_tree",
        "broadleaf_evergreen_shrub",
        "broadleaf_deciduous_temperate_shrub",
        "broadleaf_deciduous_boreal_shrub",
        "c3_arctic_grass",
        "c3_non-arctic_grass",
        "c4_grass",
        "unmanaged_c3_crop",
        "unmanaged_c3_irrigated",
        "temperate_corn",
        "irrigated_temperate_corn",
        "spring_wheat",
        "irrigated_spring_wheat",
        "winter_wheat",
        "irrigated_winter_wheat",
        "soybean",
        "irrigated_soybean",
        "barley",
        "irrigated_barley",
        "winter_barley",
        "irrigated_winter_barley",
        "rye",
        "irrigated_rye",
        "winter_rye",
        "irrigated_winter_rye",
        "cassava",
        "irrigated_cassava",
        "citrus",
        "irrigated_citrus",
        "cocoa",
        "irrigated_cocoa",
        "coffee",
        "irrigated_coffee",
        "cotton",
        "irrigated_cotton",
        "datepalm",
        "irrigated_datepalm",
        "foddergrass",
        "irrigated_foddergrass",
        "grapes",
        "irrigated_grapes",
        "groundnuts",
        "irrigated_groundnuts",
        "millet",
        "irrigated_millet",
        "oilpalm",
        "irrigated_oilpalm",
        "potatoes",
        "irrigated_potatoes",
        "pulses",
        "irrigated_pulses",
        "rapeseed",
        "irrigated_rapeseed",
        "rice",
        "irrigated_rice",
        "sorghum",
        "irrigated_sorghum",
        "sugarbeet",
        "irrigated_sugarbeet",
        "sugarcane",
        "irrigated_sugarcane",
        "sunflower",
        "irrigated_sunflower",
        "miscanthus",
        "irrigated_miscanthus",
        "switchgrass",
        "irrigated_switchgrass",
        "tropical_corn",
        "irrigated_tropical_corn",
        "tropical_soybean",
        "irrigated_tropical_soybean",
    ]
    return pftlist

#EarthStat_FAO_croplist(MIRCA2000)
def ESFAO_croplist():
    ncar_earthstatfao_crops = [
        'wheat',
        'maize',
        'rice',
        'barley',
        'rye',
        'millet',
        'sorghum',
        'soybean',
        'sunflower',
        'potato',
        'cassava',
        'sugarcane',
        'sugarbeet',
        'oilpalm',
        'rapeseed',
        'groundnut',
        'pulses',
        'citrus',
        'datepalm',
        'grape',
        'cotton',
        'cocoa',
        'coffee',
        'othersperennial',
        'foddergrass',
        'othersannual',
        'fibrecrops',
        'allcrops'
        ]
    return ncar_earthstatfao_crops

# Get CLM ivt number corresponding to a given name
def ivt_str2int(ivt_str):
    pftlist = define_pftlist()
    if isinstance(ivt_str, str):
        ivt_int = pftlist.index(ivt_str)
    elif isinstance(ivt_str, list) or isinstance(ivt_str, np.ndarray):
        ivt_int = [ivt_str2int(x) for x in ivt_str]
        if isinstance(ivt_str, np.ndarray):
            ivt_int = np.array(ivt_int)
    else:
        raise RuntimeError(
            f"Update ivt_str_to_int() to handle input of type {type(ivt_str)} (if possible)"
        )

    return ivt_int


# Get CLM ivt name corresponding to a given number
def ivt_int2str(ivt_int):
    pftlist = define_pftlist()
    if np.issubdtype(type(ivt_int), np.integer) or int(ivt_int) == ivt_int:
        ivt_str = pftlist[int(ivt_int)]
    elif isinstance(ivt_int, list) or isinstance(ivt_int, np.ndarray):
        ivt_str = [ivt_int2str(x) for x in ivt_int]
        if isinstance(ivt_int, np.ndarray):
            ivt_str = np.array(ivt_str)
    elif isinstance(ivt_int, float):
        raise RuntimeError("List indices must be integers")
    else:
        raise RuntimeError(
            f"Update ivt_str_to_int() to handle input of type {type(ivt_int)} (if possible)"
        )

    return ivt_str

def safer_timeslice(ds, timeSlice, timeVar="time"):
    try:
        ds = ds.sel({timeVar: timeSlice})
    except:
        # If the issue might have been slicing using strings, try to fall back to integer slicing
        if (
            isinstance(timeSlice.start, str)
            and isinstance(timeSlice.stop, str)
            and len(timeSlice.start.split("-")) == 3
            and timeSlice.start.split("-")[1:] == ["01", "01"]
            and len(timeSlice.stop.split("-")) == 3
            and (
                timeSlice.stop.split("-")[1:] == ["12", "31"]
                or timeSlice.stop.split("-")[1:] == ["01", "01"]
            )
        ):
            fileyears = np.array([x.year for x in ds.time.values])
            if len(np.unique(fileyears)) != len(fileyears):
                print("Could not fall back to integer slicing of years: Time axis not annual")
                raise
            yStart = int(timeSlice.start.split("-")[0])
            yStop = int(timeSlice.stop.split("-")[0])
            where_in_timeSlice = np.where((fileyears >= yStart) & (fileyears <= yStop))[0]
            ds = ds.isel({timeVar: where_in_timeSlice})
        else:
            print(f"Could not fall back to integer slicing for timeSlice {timeSlice}")
            raise

    return ds
# Function to drop unwanted variables in preprocessing of open_mfdataset(), making sure to NOT drop any unspecified variables that will be useful in gridding. Also adds vegetation type info in the form of a DataArray of strings.
# Also renames "pft" dimension (and all like-named variables, e.g., pft1d_itype_veg_str) to be named like "patch". This can later be reversed, for compatibility with other code, using patch2pft().
def mfdataset_preproc(ds, vars_to_import, vegtypes_to_import, timeSlice, add_time_axis_vars):
    # Rename "pft" dimension and variables to "patch", if needed
    if "pft" in ds.dims:
        pattern = re.compile("pft.*1d")
        matches = [x for x in list(ds.keys()) if pattern.search(x) != None]
        pft2patch_dict = {"pft": "patch"}
        for m in matches:
            pft2patch_dict[m] = m.replace("pft", "patch").replace("patchs", "patches")
        ds = ds.rename(pft2patch_dict)

    derived_vars = []
    if vars_to_import != None:
        # Split vars_to_import into variables that are vs. aren't already in ds
        derived_vars = [v for v in vars_to_import if v not in ds]
        present_vars = [v for v in vars_to_import if v in ds]
        vars_to_import = present_vars

        # Get list of dimensions present in variables in vars_to_import.
        dimList = []
        for thisVar in vars_to_import:
            # list(set(x)) returns a list of the unique items in x
            dimList = list(set(dimList + list(ds.variables[thisVar].dims)))

        # Get any _1d variables that are associated with those dimensions. These will be useful in gridding. Also, if any dimension is "pft", set up to rename it and all like-named variables to "patch"
        onedVars = get_useful_1d_vars(ds, dimList)

        # Add dimensions and _1d variables to vars_to_import
        vars_to_import = list(set(vars_to_import + list(ds.dims) + onedVars))

        # Add any _bounds variables
        bounds_vars = []
        for v in vars_to_import:
            bounds_var = v + "_bounds"
            if bounds_var in ds:
                bounds_vars = bounds_vars + [bounds_var]
        vars_to_import = vars_to_import + bounds_vars

        # Get list of variables to drop
        varlist = list(ds.variables)
        vars_to_drop = list(np.setdiff1d(varlist, vars_to_import))

        # Drop them
        ds = ds.drop_vars(vars_to_drop)

    # Add time axis to useful 1-d variables, if needed
    if add_time_axis_vars:
        if isinstance(add_time_axis_vars, str):
            add_time_axis_vars = [add_time_axis_vars]
        elif not isinstance(add_time_axis_vars, list):
            add_time_axis_vars = get_useful_1d_vars(ds, ds.dims)
        for var in add_time_axis_vars:
            da = ds[var]
            if "time" in da.dims:
                continue
            ds[var] = da.expand_dims({"time": ds["time"]})

    # Add vegetation type info
    if "patches1d_itype_veg" in list(ds):
        this_pftlist = define_pftlist()
        get_patch_ivts(
            ds, this_pftlist
        )  # Includes check of whether vegtype changes over time anywhere
        vegtype_da = get_vegtype_str_da(this_pftlist)
        patches1d_itype_veg_str = vegtype_da.values[
            ds.isel(time=0).patches1d_itype_veg.values.astype(int)
        ]
        npatch = len(patches1d_itype_veg_str)
        patches1d_itype_veg_str = xr.DataArray(
            patches1d_itype_veg_str,
            coords={"patch": np.arange(0, npatch)},
            dims=["patch"],
            name="patches1d_itype_veg_str",
        )
        ds = xr.merge([ds, vegtype_da, patches1d_itype_veg_str])

    # Restrict to veg. types of interest, if any
    if vegtypes_to_import != None:
        ds = xr_flexsel(ds, vegtype=vegtypes_to_import)

    # Restrict to time slice, if any
    if timeSlice:
        ds = safer_timeslice(ds, timeSlice)

    # Finish import
    ds = xr.decode_cf(ds, decode_times=True)

    # Compute derived variables
    for v in derived_vars:
        if v == "HYEARS" and "HDATES" in ds and ds.HDATES.dims == ("time", "mxharvests", "patch"):
            yearList = np.array([np.float32(x.year - 1) for x in ds.time.values])
            hyears = ds["HDATES"].copy()
            hyears.values = np.tile(
                np.expand_dims(yearList, (1, 2)), (1, ds.sizes["mxharvests"], ds.sizes["patch"])
            )
            with np.errstate(invalid="ignore"):
                is_le_zero = ~np.isnan(ds.HDATES.values) & (ds.HDATES.values <= 0)
            hyears.values[is_le_zero] = ds.HDATES.values[is_le_zero]
            hyears.values[np.isnan(ds.HDATES.values)] = np.nan
            hyears.attrs["long_name"] = "DERIVED: actual crop harvest years"
            hyears.attrs["units"] = "year"
            ds["HYEARS"] = hyears

    return ds
    
# Import a dataset that can be spread over multiple files, only including specified variables and/or vegetation types and/or timesteps, concatenating by time. DOES actually read the dataset into memory, but only AFTER dropping unwanted variables and/or vegetation types.
def import_ds(
    filelist,
    myVars=None,
    myVegtypes=None,
    timeSlice=None,
    myVars_missing_ok=[],
    only_active_patches=False,
    rename_lsmlatlon=False,
    chunks=None,
    add_time_axis_vars=None,
):
    # Convert myVegtypes here, if needed, to avoid repeating the process each time you read a file in xr.open_mfdataset().
    if myVegtypes != None:
        if not isinstance(myVegtypes, list):
            myVegtypes = [myVegtypes]
        if isinstance(myVegtypes[0], str):
            myVegtypes = vegtype_str2int(myVegtypes)

    # Same for these variables.
    if myVars != None:
        if not isinstance(myVars, list):
            myVars = [myVars]
    if myVars_missing_ok:
        if not isinstance(myVars_missing_ok, list):
            myVars_missing_ok = [myVars_missing_ok]

    # Make sure lists are actually lists
    if not isinstance(filelist, list):
        filelist = [filelist]
    if not isinstance(myVars_missing_ok, list):
        myVars_missing_ok = [myVars_missing_ok]

    # Remove files from list if they don't contain requested timesteps.
    # timeSlice should be in the format slice(start,end[,step]). start or end can be None to be unbounded on one side. Note that the standard slice() documentation suggests that only elements through end-1 will be selected, but that seems not to be the case in the xarray implementation.
    if timeSlice:
        new_filelist = []
        for file in sorted(filelist):
            filetime = xr.open_dataset(file).time
            filetime_sel = safer_timeslice(filetime, timeSlice)
            include_this_file = filetime_sel.size
            if include_this_file:
                new_filelist.append(file)

            # If you found some matching files, but then you find one that doesn't, stop going through the list.
            elif new_filelist:
                break
        if not new_filelist:
            raise RuntimeError(f"No files found in timeSlice {timeSlice}")
        filelist = new_filelist

    # The xarray open_mfdataset() "preprocess" argument requires a function that takes exactly one variable (an xarray.Dataset object). Wrapping mfdataset_preproc() in this lambda function allows this. Could also just allow mfdataset_preproc() to access myVars and myVegtypes directly, but that's bad practice as it could lead to scoping issues.
    mfdataset_preproc_closure = lambda ds: mfdataset_preproc(ds, myVars, myVegtypes, timeSlice, add_time_axis_vars)

    # Import
    if isinstance(filelist, list) and len(filelist) == 1:
        filelist = filelist[0]
    if isinstance(filelist, list):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=DeprecationWarning)
            if importlib.find_loader("dask") is None:
                raise ModuleNotFoundError(
                    "You have asked xarray to import a list of files as a single Dataset using"
                    " open_mfdataset(), but this requires dask, which is not available.\nFile"
                    f" list: {filelist}"
                )
        this_ds = xr.open_mfdataset(
            sorted(filelist),
            data_vars="minimal",
            preprocess=mfdataset_preproc_closure,
            compat="override",
            coords="all",
            concat_dim="time",
            combine="nested",
            chunks=chunks,
        )
    elif isinstance(filelist, str):
        this_ds = xr.open_dataset(filelist, chunks=chunks)
        this_ds = mfdataset_preproc(
            this_ds,
            myVars,
            myVegtypes,
            timeSlice,
            add_time_axis_vars=None,  # If only opening one file, no use to adding time axis
        )
        this_ds = this_ds.compute()

    # Include only active patches (or whatever)
    if only_active_patches:
        is_active = this_ds.patches1d_active.values
        p_active = np.where(is_active)[0]
        this_ds_active = this_ds.isel(patch=p_active)

    # Warn and/or error about variables that couldn't be imported or derived
    if myVars:
        missing_vars = [v for v in myVars if v not in this_ds]
        ok_missing_vars = [v for v in missing_vars if v in myVars_missing_ok]
        bad_missing_vars = [v for v in missing_vars if v not in myVars_missing_ok]
        if ok_missing_vars:
            print(
                "Could not import some variables; either not present or not deriveable:"
                f" {ok_missing_vars}"
            )
        if bad_missing_vars:
            raise RuntimeError(
                "Could not import some variables; either not present or not deriveable:"
                f" {bad_missing_vars}"
            )

    if rename_lsmlatlon:
        if "lsmlat" in this_ds.dims:
            this_ds = this_ds.rename({"lsmlat": "lat"})
        if "lsmlon" in this_ds.dims:
            this_ds = this_ds.rename({"lsmlon": "lon"})

    return this_ds

# Convert a longitude axis that's 0 to 360 around the prime meridian to one that's -180 to 180 around the international date line. If you pass in a Dataset or DataArray, the "lon" coordinates will be changed and the axis and data rolled---i.e., maps will be centered on the prime meridian, plus or minus any offset of your gridcell centers. Otherwise, this assumes you're passing in numeric data, and no rolling takes place.
def lon_pm2idl(lons_in, fail_silently=False):
    def check_ok(tmp, fail_silently):
        msg = ""
        if np.any(tmp < 0):
            msg = f"Minimum longitude is already < 0 ({np.min(tmp)})"
        elif np.any(tmp > 360):
            msg = f"Maximum longitude is > 360 ({np.max(tmp)})"

        if msg == "":
            return True
        elif fail_silently:
            return False
        else:
            raise ValueError(msg)

    def do_it(tmp):
        tmp = np.mod((tmp + 180), 360) - 180
        return tmp

    if isinstance(lons_in, (xr.DataArray, xr.Dataset)):
        if not check_ok(lons_in.lon.values, fail_silently):
            return lons_in
        lons_out = lons_in
        lons_out = lons_out.assign_coords(lon=do_it(lons_in.lon.values))
        lons_out = make_lon_increasing(lons_out)
    else:
        if not check_ok(lons_in, fail_silently):
            return lons_in
        lons_out = do_it(lons_in)
        if not is_strictly_increasing(lons_out):
            print(
                "WARNING: You passed in numeric longitudes to lon_pm2idl() and these have been"
                " converted, but they're not strictly increasing."
            )
        print(
            "To assign the new longitude coordinates to an Xarray object, use"
            " xarrayobject.assign_coordinates()! (Pass the object directly in to lon_pm2idl() in"
            " order to suppress this message.)"
        )

    return lons_out

def is_this_vegtype(this_vegtype, this_filter, this_method):
    # Make sure data type of this_vegtype is acceptable
    if isinstance(this_vegtype, float) and int(this_vegtype) == this_vegtype:
        this_vegtype = int(this_vegtype)
    data_type_ok = lambda x: isinstance(x, str) or isinstance(x, int) or isinstance(x, np.int64)
    ok_input = True
    if not data_type_ok(this_vegtype):
        if isinstance(this_vegtype, xr.core.dataarray.DataArray):
            this_vegtype = this_vegtype.values
        if isinstance(this_vegtype, (list, np.ndarray)):
            if len(this_vegtype) == 1 and data_type_ok(this_vegtype[0]):
                this_vegtype = this_vegtype[0]
            elif data_type_ok(this_vegtype[0]):
                raise TypeError(
                    "is_this_vegtype(): this_vegtype must be a single string or integer, not a list"
                    " of them. Did you mean to call is_each_vegtype() instead?"
                )
            else:
                ok_input = False
        else:
            ok_input = False
    if not ok_input:
        raise TypeError(
            "is_this_vegtype(): First argument (this_vegtype) must be a string or integer, not"
            f" {type(this_vegtype)}"
        )

    # Make sure data type of this_filter is acceptable
    if not np.iterable(this_filter):
        raise TypeError(
            "is_this_vegtype(): Second argument (this_filter) must be iterable (e.g., a list), not"
            f" {type(this_filter)}"
        )

    # Perform the comparison
    if this_method == "ok_contains":
        return any(n in this_vegtype for n in this_filter)
    elif this_method == "notok_contains":
        return not any(n in this_vegtype for n in this_filter)
    elif this_method == "ok_exact":
        return any(n == this_vegtype for n in this_filter)
    elif this_method == "notok_exact":
        return not any(n == this_vegtype for n in this_filter)
    else:
        raise ValueError(f"Unknown comparison method: '{this_method}'")


# Get boolean list of whether each vegetation type in list is a managed crop
"""
    this_vegtypelist: The list of vegetation types whose members you want to
                      test.
    this_filter:      The list of strings against which you want to compare
                      each member of this_vegtypelist.
    this_method:      How you want to do the comparison. See is_this_vegtype().
"""


def is_each_vegtype(this_vegtypelist, this_filter, this_method):
    if isinstance(this_vegtypelist, xr.DataArray):
        this_vegtypelist = this_vegtypelist.values

    return [is_this_vegtype(x, this_filter, this_method) for x in this_vegtypelist]


# Helper function to check that a list is strictly increasing
def is_strictly_increasing(L):
    # https://stackoverflow.com/a/4983359/2965321
    return all(x < y for x, y in zip(L, L[1:]))

# Ensure that longitude axis coordinates are monotonically increasing
def make_lon_increasing(xr_obj):
    if not "lon" in xr_obj.dims:
        return xr_obj

    lons = xr_obj.lon.values
    if is_strictly_increasing(lons):
        return xr_obj

    shift = 0
    while not is_strictly_increasing(lons) and shift < lons.size:
        shift = shift + 1
        lons = np.roll(lons, 1, axis=0)
    if not is_strictly_increasing(lons):
        raise RuntimeError("Unable to rearrange longitude axis so it's monotonically increasing")

    return xr_obj.roll(lon=shift, roll_coords=True)

# List (strings) of managed crops in CLM.
def define_mgdcrop_list():
    notcrop_list = ["tree", "grass", "shrub", "unmanaged", "not_vegetated"]
    defined_pftlist = define_pftlist()
    is_crop = is_each_vegtype(defined_pftlist, notcrop_list, "notok_contains")
    return [defined_pftlist[i] for i, x in enumerate(is_crop) if x]


# Convert list of vegtype strings to integer index equivalents.
def vegtype_str2int(vegtype_str, vegtype_mainlist=None):
    convert_to_ndarray = not isinstance(vegtype_str, np.ndarray)
    if convert_to_ndarray:
        vegtype_str = np.array(vegtype_str)

    if isinstance(vegtype_mainlist, xr.Dataset):
        vegtype_mainlist = vegtype_mainlist.vegtype_str.values
    elif isinstance(vegtype_mainlist, xr.DataArray):
        vegtype_mainlist = vegtype_mainlist.values
    elif vegtype_mainlist == None:
        vegtype_mainlist = define_pftlist()
    if not isinstance(vegtype_mainlist, list) and isinstance(vegtype_mainlist[0], str):
        if isinstance(vegtype_mainlist, list):
            raise TypeError(
                f"Not sure how to handle vegtype_mainlist as list of {type(vegtype_mainlist[0])}"
            )
        else:
            raise TypeError(
                f"Not sure how to handle vegtype_mainlist as type {type(vegtype_mainlist[0])}"
            )

    if vegtype_str.shape == ():
        indices = np.array([-1])
    else:
        indices = np.full(len(vegtype_str), -1)
    for v in np.unique(vegtype_str):
        indices[np.where(vegtype_str == v)] = vegtype_mainlist.index(v)
    if convert_to_ndarray:
        indices = [int(x) for x in indices]
    return indices