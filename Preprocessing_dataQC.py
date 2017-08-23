
# coding: utf-8

# In[ ]:

# Libraries and tools
import os
from os.path import join
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink

# FSL set up- change default file output type
from nipype.interfaces.fsl import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

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


# Workflow build
preprocflow = Workflow(name='preprocflow')

preprocflow.connect([(infosource, selectfiles, [('subjid', 'subjid')]),
                     (selectfiles, reorientFunc, [('func','in_file')]),
                     (infosource, fssource, [('subjid','subject_id')]),
                     (fssource, resample, [('brainmask','in_file')]),
                     (resample, reorientAnat, [('out_file','in_file')]),
                     (reorientAnat, binarize, [('out_file','in_file')]),
                     (reorientFunc, slicetime, [('out_file','in_file')]),
                     (slicetime, realignmc, [('slice_time_corrected_file','in_file')]),
                     (reorientAnat, coregflt, [('out_file','reference')]),
                     (realignmc, coregflt, [('out_file','in_file')]),
                     (realignmc, coregflt2, [('out_file','in_file')]),
                     (coregflt, coregflt2, [('out_matrix_file','in_matrix_file')]),
                     (reorientAnat, coregflt2, [('out_file','reference')]),
                     (binarize, art, [('binary_file','mask_file')]),
                     (coregflt2, art, [('out_file','realigned_files')]),
                     (realignmc, art, [('par_file','realignment_parameters')]),
                     (coregflt, make_coreg_img, [('out_file','epi')]),
                     (reorientAnat, make_coreg_img, [('out_file','anat')]),
                     (coregflt, make_checkmask_img, [('out_file','epi')]),
                     (binarize, make_checkmask_img, [('binary_file','brainmask')]),
                     (make_coreg_img, datasink, [('coreg_file','coreg_image')]),
                     (make_checkmask_img, datasink, [('maskcheck_file','checkmask_image')]),
                     (coregflt2, datasink, [('out_file','coreg_func')]),
                     #(coregflt, datasink, [('out_file','coreg_firstvol')]),
                     (reorientAnat, datasink, [('out_file','reoriented_anat')]),
                     (binarize, datasink, [('binary_file','binarized_anat')]),
                     (art, datasink, [('plot_files','art_plot')]), 
                     (art, datasink, [('outlier_files','art_outliers')]),
                     (realignmc, datasink, [('par_file','mcflirt_displacement')])
                     ])

preprocflow.base_dir = join(wkflow_dir)
preprocflow.write_graph(graph2use='flat')
preprocflow.run('MultiProc', plugin_args={'n_procs': 1})


# In[ ]:



