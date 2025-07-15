def get_corrplotname(params):
    corrplotname = f'../plots/cross_treecorr_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: corrplotname+=f'class{params['gclass']}'
    elif params['gclass'] == 3: corrplotname+=f'class{params['gclass']}'
    # Add deflection
    if params['deflection'] == 'low': corrplotname+=f'_def{params['deflection']}{int(params['def_thresh'])}'
    elif params['deflection'] == 'high': corrplotname+=f'_def{params['deflection']}{int(params['def_thresh'])}'
    # Add czmax
    if params['cz_max'] is not None:
        corrplotname += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    corrplotname += f'_dec{int(params['dec_max'])}'
    # Add format
    corrplotname += '.png'
    print(f'Save correlation plots to: {corrplotname}')

    return corrplotname

def get_skyplotname(params):
    skyplotname = f'../plots/skyplot_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: skyplotname+=f'class{params['gclass']}'
    elif params['gclass'] == 3: skyplotname+=f'class{params['gclass']}'
    # Add deflection
    if params['deflection'] == 'low': skyplotname+=f'_def{params['deflection']}{int(params['def_thresh'])}'
    elif params['deflection'] == 'high': skyplotname+=f'_def{params['deflection']}{int(params['def_thresh'])}'
    # Add czmax
    if params['cz_max'] is not None:
        skyplotname += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    skyplotname += f'_dec{int(params['dec_max'])}'
    # Add format
    skyplotname += '.png'
    print(f'Save sky coverage plots to: {skyplotname}')

    return skyplotname

def get_filename(params):
    filename = f'../data/int{str(int(params["maxsep"]))}_K_nq{params["nquant"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: filename+=f'class{params['gclass']}'
    elif params['gclass'] == 3: filename+=f'class{params['gclass']}'
    # Add deflection
    if params['deflection'] == 'low': filename+=f'_def{params['deflection']}{int(params['def_thresh'])}'
    elif params['deflection'] == 'high': filename+=f'_def{params['deflection']}{int(params['def_thresh'])}'
    # Add czmax
    if params['cz_max'] is not None:
        filename += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    filename += f'_dec{int(params['dec_max'])}'
    # Add format
    filename += '.npz'  

    return filename

def get_filecorrname(params):
    filecorrname = f'../data/cross_treecorr_nq{params["nquant"]}_nmult{params["nmult"]}_nbs{params["nbootstrap"]}_{params["sample"]}'
    # Add class
    if params['gclass'] == 2: filecorrname+=f'class{params["gclass"]}'
    elif params['gclass'] == 3: filecorrname+=f'class{params["gclass"]}'
    # Add deflection
    if params['deflection'] == 'low': filecorrname+=f'_def{params["deflection"]}{int(params["def_thresh"])}'
    elif params['deflection'] == 'high': filecorrname+=f'_def{params["deflection"]}{int(params["def_thresh"])}'
    # Add czmax
    if params['cz_max'] is not None:
        filecorrname += f'_cz{int(params["cz_min"])}-{int(params["cz_max"])}'
    filecorrname += f'_dec{int(params['dec_max'])}'
    # Add format
    filecorrname += '.npz'

    return filecorrname
