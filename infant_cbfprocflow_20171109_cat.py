
# coding: utf-8

# In[ ]:

from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode

from nipype.interfaces.slicer.registration import brainsresample
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.freesurfer.preprocess import MRIConvert, ApplyVolTransform
from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.interfaces.fsl.preprocess import FLIRT, SUSAN
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.spm.preprocess import Smooth

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12/toolbox')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

#other study-specific variables
#project_home = '/Volumes/active-19/BABIES/BABIES_perfusion'
project_home = '/Users/catcamacho/Dropbox/Projects/infant_ASL/proc'
raw_dir = project_home + '/Raw'
subjects_list = open(project_home + '/Misc/Subjects.txt').read().splitlines()
#subjects_list = ['012']
output_dir = project_home + '/Proc'
wkflow_dir = project_home + '/Workflows'
template = project_home + '/Templates/T2wtemplate_2mm.nii'

#Population specific variables for ASL
nex_asl = 3 #number of excitations from the 3D ASL scan parameters
inversion_efficiency = 0.8 #from GE
background_supp_eff = 0.75 #from GE
efficiency = inversion_efficiency * background_supp_eff 
T1_blood = 1.6 #T1 of blood in seconds(1.6s at 3T and 1.4s at 1.5T)
sat_time = 2 #in seconds, from GE
partition_coeff = 0.9 #whole brain average in ml/g
scaling_factor = 32 #scaling factor, can be taken from PW dicom header at position 0043,107f (corresponds to #coils?)
postlabel_delay = 1.525 #post label delay in seconds
labeling_time = 1.450 #labeling time in seconds
T1_tissue = 1.2 #estimated T1 of grey matter in seconds
TR = 4.844 #repetition time

smoothing_kernel = 4 # in mm


# In[ ]:

## File handling nodes

# Select subjects
infosource = Node(IdentityInterface(fields=['subjid', 'volume']),
                  name='infosource')
infosource.iterables = [('subjid', subjects_list)]


# SelectFiles
templates = {'pw_volume': raw_dir + '/{subjid}-C-T1/pw.nii',
             'anat_volume': raw_dir + '/{subjid}-C-T1/processed_t2.nii',
             'pd_volume': raw_dir + '/{subjid}-C-T1/pd.nii'}
selectfiles = Node(SelectFiles(templates), name='selectfiles')


# Datasink
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir
# DataSink output substitutions (for ease of folder naming)
substitutions = [('_subjid_', ''),
                ('volume_',''),
                ('_reoriented',''),
                ('_warped','')]
datasink.inputs.substitutions = substitutions


# In[ ]:

## File Processing nodes

# convert files to nifti
mri_convert = Node(MRIConvert(out_type='nii',
                                conform_size=2,
                                crop_size= (128, 128, 128)), 
                   name='mri_convert')

# reorient data for consistency
reorient_anat = Node(Reorient2Std(output_type='NIFTI'),
                     name='reorient_anat')

# reorient data for consistency
reorient_pd = Node(Reorient2Std(output_type='NIFTI'),
                   name='reorient_pd')

# reorient data for consistency
reorient_pw = Node(Reorient2Std(output_type='NIFTI'),
                   name='reorient_pw')


# Binarize -  binarize and dilate image to create a brainmask
binarize = Node(Binarize(min=0.5,
                         dilate=2,
                         erode=1,
                         out_type='nii'),
                name='binarize')

    

reg2anat = Node(FLIRT(out_matrix_file = 'xform.mat'), name = 'reg2anat')
applyxform = Node(FLIRT(apply_xfm = True), name = 'applyxform')

# Mask brain in pw and pd volumes
applyMask_pd = Node(ApplyMask(output_type='NIFTI'), 
                    name='applyMask_pd')

applyMask_pw = Node(ApplyMask(output_type='NIFTI'), 
                    name='applyMask_pw')

# N3 bias correction using MINC tools, will be necessary for babies
# nu_correct = Node(fs.MNIBiasCorrection(), name='nu_correct')


# In[ ]:

# Data QC nodes
def create_coreg_plot(epi,anat):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    coreg_filename='coregistration.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'coregistration to anatomy')
    display.add_edges(anat)
    display.savefig(coreg_filename) 
    display.close()
    coreg_file = os.path.abspath(coreg_filename)
    
    return(coreg_file)

def check_mask_coverage(epi,brainmask):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    maskcheck_filename='maskcheck.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'brainmask coverage')
    display.add_contours(brainmask,levels=[.5], colors='r')
    display.savefig(maskcheck_filename)
    display.close()
    maskcheck_file = os.path.abspath(maskcheck_filename)

    return(maskcheck_file)

make_coreg_img = Node(name='make_coreg_img',
                      interface=Function(input_names=['epi','anat'],
                                         output_names=['coreg_file'],
                                         function=create_coreg_plot))

make_checkmask_img = Node(name='make_checkmask_img',
                      interface=Function(input_names=['epi','brainmask'],
                                         output_names=['maskcheck_file'],
                                         function=check_mask_coverage))


# In[ ]:

# Create a flow for preprocessing anat + asl volumes 
preprocflow = Workflow(name='preprocflow')

# Connect all components of the preprocessing workflow
preprocflow.connect([(infosource, selectfiles, [('subjid', 'subjid')]),
                     (selectfiles, mri_convert, [('anat_volume', 'in_file')]),
                     (mri_convert, reorient_anat, [('out_file', 'in_file')]),    
                     (selectfiles, reorient_pw, [('pw_volume', 'in_file')]), 
                     (selectfiles, reorient_pd, [('pd_volume', 'in_file')]),
                     (reorient_anat, reg2anat, [('out_file', 'reference')]),
                     (reorient_pw, reg2anat, [('out_file', 'in_file')]),
                     (reg2anat, applyxform, [('out_matrix_file','in_matrix_file')]),
                     (reorient_anat, applyxform, [('out_file','reference')]),
                     (reorient_pd, applyxform, [('out_file','in_file')]),
                     (reorient_anat, binarize, [('out_file','in_file')]),
                     (applyxform, applyMask_pd, [('out_file','in_file')]),
                     (binarize, applyMask_pd, [('binary_file','mask_file')]),
                     (reg2anat, applyMask_pw, [('out_file','in_file')]),
                     (binarize, applyMask_pw, [('binary_file','mask_file')]),
                     (reg2anat, make_coreg_img, [('out_file','epi')]),
                     (reorient_anat, make_coreg_img, [('out_file','anat')]),
                     (make_coreg_img, datasink, [('coreg_file','coreg_check')]),
                     (reg2anat, make_checkmask_img, [('out_file','epi')]),
                     (binarize, make_checkmask_img, [('binary_file','brainmask')]),
                     (make_checkmask_img, datasink, [('maskcheck_file','masked_check')]),
                     (applyMask_pd, datasink, [('out_file','masked_pd')]),
                     (applyMask_pw, datasink, [('out_file','masked_pw')]),
                     (reorient_anat, datasink, [('out_file', 'reorient_anat')])
                    ])
preprocflow.base_dir = wkflow_dir
preprocflow.write_graph(graph2use='flat')
preprocflow.run('MultiProc', plugin_args={'n_procs': 2})


# In[ ]:

## File handling nodes for CBF proc

# Select subjects
cbfinfosource = Node(IdentityInterface(fields=['subjid']),
                  name='cbfinfosource')
cbfinfosource.iterables = [('subjid', subjects_list)]


# SelectFiles
templates = {'pw_volume': output_dir + '/masked_pw/{subjid}/pw_flirt_masked.nii',
            'pd_volume': output_dir + '/masked_pd/{subjid}/pd_flirt_masked.nii',
            'anat_volume': output_dir + '/reorient_anat/{subjid}/processed_t2_out.nii'}
cbfselectfiles = Node(SelectFiles(templates), name='cbfselectfiles')

# Datasink
cbfdatasink = Node(DataSink(), name='cbfdatasink')
cbfdatasink.inputs.base_directory = output_dir
cbfdatasink.inputs.container = output_dir
# DataSink output substitutions (for ease of folder naming)
substitutions = [('_subjid_', '')]
cbfdatasink.inputs.substitutions = substitutions


# In[ ]:

## Custom functions

#quantify CBF from PW volume (Alsop MRIM 2015 method)
def quantify_cbf_alsop(pw_volume,pd_volume,sat_time,postlabel_delay,T1_blood,labeling_time,efficiency,partition_coeff,TR,T1_tissue,scaling_factor,nex_asl):
    import os
    import nibabel as nib
    from numpy import exp
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    # set variables
    pw_nifti1 = nib.load(pw_volume)
    pw_data = pw_nifti1.get_data()
    pw_data = pw_data.astype(float)
    pd_nifti1 = nib.load(pd_volume)
    pd_data = pd_nifti1.get_data()
    pd_data = pd_data.astype(float)
    conversion = 6000 #to convert values from mL/g/s to mL/100g/min
    pd_factor = 1/(1-exp((-1*TR)/T1_tissue))
    
    cbf_numerator = conversion*partition_coeff*pw_data*exp(postlabel_delay/T1_blood)
    cbf_denominator = sat_time*efficiency*T1_blood*scaling_factor*nex_asl*pd_data*pd_factor*(1-exp((-1*labeling_time)/T1_blood))
    cbf_data = cbf_numerator/cbf_denominator
    
    cbf_volume = nib.Nifti1Image(cbf_data, pw_nifti1.affine)
    nib.save(cbf_volume, 'alsop_cbf.nii')
    cbf_path = os.path.abspath('alsop_cbf.nii')
    return cbf_path

quant_cbf_alsop = Node(name='quant_cbf_alsop',
                interface=Function(input_names=['pw_volume','pd_volume',
                                                'sat_time','postlabel_delay',
                                                'T1_blood','labeling_time',
                                                'efficiency','partition_coeff',
                                                'TR','T1_tissue','scaling_factor',
                                                'nex_asl'],
                                  output_names=['cbf_volume'],
                                  function=quantify_cbf_alsop))
quant_cbf_alsop.inputs.sat_time=sat_time
quant_cbf_alsop.inputs.postlabel_delay=postlabel_delay
quant_cbf_alsop.inputs.T1_blood=T1_blood
quant_cbf_alsop.inputs.labeling_time=labeling_time
quant_cbf_alsop.inputs.efficiency=efficiency
quant_cbf_alsop.inputs.partition_coeff=partition_coeff
quant_cbf_alsop.inputs.TR=TR
quant_cbf_alsop.inputs.T1_tissue=T1_tissue
quant_cbf_alsop.inputs.scaling_factor=scaling_factor
quant_cbf_alsop.inputs.nex_asl=nex_asl

# Calculate Brightness Threshold for SUSAN

# Brightness threshold should be 0.75 * the contrast between the median brain intensity and the background
def calc_brightness_threshold(func_vol):
    import nibabel as nib
    from numpy import median, where
    
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    func_nifti1 = nib.load(func_vol)
    func_data = func_nifti1.get_data()
    func_data = func_data.astype(float)
    
    brain_values = where(func_data > 0)
    median_thresh = median(brain_values)
    brightness_threshold = 0.75 * median_thresh
    return(brightness_threshold)


# In[ ]:

## Normalizing data for first and second level analysis

# Calculate brightness threshold
calc_bright_thresh = Node(Function(input_names=['func_vol'],
                                   output_names=['brightness_threshold'],
                                   function=calc_brightness_threshold), 
                          name='calc_bright_thresh')

# Smooth parameter estimates- input brightness_threshold and in_file; output smoothed_file
#smooth = Node(SUSAN(fwhm=smoothing_kernel,brightness_threshold=75), 
#              name='smooth')

smooth = Node(Smooth(fwhm=[4,4,4], 
                     implicit_masking=True), 
              name='smooth')

# Register subject's anatomy to the template
linearReg = Node(FLIRT(output_type='NIFTI',
                       reference=template),
                 name='linearReg')

## Register CBF vol to MNI space
# Volume Transformation - transform the cbf volume into MNI space
warpCBF = Node(ApplyVolTransform(inverse=False,
                                 target_file=template),
               name='warpCBF')


# In[ ]:

# create a flow for quantifying CBF and warping to MNI space.
cbfprocflow = Workflow(name='cbfprocflow')

# connect the nodes
cbfprocflow.connect([(cbfinfosource, cbfselectfiles, [('subjid', 'subjid')]),
                     (cbfselectfiles, quant_cbf_alsop, [('pw_volume', 'pw_volume')]),
                     (cbfselectfiles, quant_cbf_alsop, [('pd_volume', 'pd_volume')]),
                     (quant_cbf_alsop, cbfdatasink, [('cbf_volume', 'alsop_quant_cbf')]),
                     (cbfselectfiles, linearReg, [('anat_volume', 'in_file')]),
                     (linearReg, cbfdatasink, [('out_file','linwarped_anat')]),
                     (linearReg, warpCBF, [('out_matrix_file', 'fsl_reg_file')]),
                     #(quant_cbf_alsop, calc_bright_thresh, [('cbf_volume','func_vol')]),
                     #(calc_bright_thresh, smooth, [('brightness_threshold','brightness_threshold')]),
                     (quant_cbf_alsop, smooth, [('cbf_volume','in_files')]),
                     (smooth, warpCBF, [('smoothed_files', 'source_file')]),
                     (warpCBF, cbfdatasink, [('transformed_file', 'warped_cbf_vol')])
                    ]),
cbfprocflow.base_dir = wkflow_dir
cbfprocflow.write_graph(graph2use='flat')
cbfprocflow.run('MultiProc', plugin_args={'n_procs': 2})

