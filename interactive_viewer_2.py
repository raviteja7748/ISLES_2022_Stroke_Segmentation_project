#!/usr/bin/env python3
"""
Clinical Interactive 2D Slice Viewer
Clean, minimal, and effective stroke lesion validation
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# Import metric calculation functions
from merger_refactored_complete import (
    calculate_dice_score, calculate_iou_score, calculate_sensitivity, 
    calculate_specificity, calculate_precision
)

class ClinicalSliceViewer:
    """Clean professional 2D slice viewer for clinical validation"""
    
    def __init__(self):
        # Initialize all attributes first
        self.available_patients = []
        self.patients_data = {}
        self.analysis_data = None
        self.using_corrected = False
        
        # Current state
        self.current_patient_idx = 0
        self.current_slice = 72
        self.show_mode = 'overlay'
        
        # Data containers
        self.current_data = {}
        self.lesion_stats = {}
        
        # UI components
        self.fig = None
        self.axes = None
        self.widgets = {}
        
        # Setup paths and data
        self.setup_paths()
        self.load_analysis_data()
        
        # Setup UI and load (only if patients available)
        if self.available_patients:
            self.setup_viewer()
            self.load_patient_data()
        else:
            print("No patients available - cannot start viewer")
        
    def setup_paths(self):
        """Setup data paths"""
        possible_dirs = [
            Path("/mnt/e/FINIAL PROJECT"),
            Path("E:/FINIAL PROJECT"), 
            Path("./"),
            Path(".")
        ]
        
        self.base_path = None
        for dir_path in possible_dirs:
            aligned_exists = (dir_path / "aligned_multimodal").exists()
            batch_exists = (dir_path / "batch_results_2").exists()
            print(f"Checking {dir_path}: aligned_multimodal={aligned_exists}, batch_results_2={batch_exists}")
            if aligned_exists and batch_exists:
                self.base_path = dir_path
                break
                
        if self.base_path is None:
            raise FileNotFoundError("Could not find project directory")
            
        self.aligned_dir = self.base_path / "aligned_multimodal"
        self.batch_dir = self.base_path / "batch_results_2"
        print(f"Using base path: {self.base_path}")
        print(f"Aligned dir: {self.aligned_dir}")
        print(f"Batch dir: {self.batch_dir}")
        
    def load_analysis_data(self):
        """Load analysis data for small lesion tracking"""
        # Use comprehensive_batch_analysis_corrected.json (prioritize corrected results)
        corrected_json = self.batch_dir / 'comprehensive_batch_analysis_corrected.json'
        batch_json = self.batch_dir / 'batch_inference_results_2.json'
        
        if corrected_json.exists():
            json_path = corrected_json
            self.using_corrected = True
            
            with open(json_path, 'r') as f:
                self.analysis_data = json.load(f)
            self.patients_data = {p['patient_num']: p for p in self.analysis_data['individual_results']}
            
        elif batch_json.exists():
            json_path = batch_json
            self.using_corrected = False
            
            with open(json_path, 'r') as f:
                self.analysis_data = json.load(f)
            self.patients_data = {p['patient_num']: p for p in self.analysis_data['results']}
        else:
            self.analysis_data = None
            self.patients_data = {}
            self.using_corrected = False
            return
            
        self.find_available_patients()
        
    def find_available_patients(self):
        """Find patients with complete data - including GT=0 cases"""
        # Get patients with predictions
        pred_files = list(self.batch_dir.glob("prediction_*.nii.gz"))
        patients_with_predictions = []
        
        for pred_file in pred_files:
            try:
                patient_part = pred_file.name.split('strokecase')[1].split('_')[0]
                patient_num = int(patient_part)
                patients_with_predictions.append(patient_num)
            except:
                continue
        
        # Get all patients with images (including GT=0 cases)
        dwi_files = list(self.aligned_dir.glob("dwi/dwi_aligned_*.nii.gz"))
        all_patients = []
        
        for dwi_file in dwi_files:
            try:
                patient_part = dwi_file.name.split('strokecase')[1].split('_')[0]
                patient_num = int(patient_part)
                all_patients.append(patient_num)
            except:
                continue
        
        # Combine both lists and sort
        self.available_patients = sorted(set(patients_with_predictions + all_patients))
        self.patients_with_predictions = set(patients_with_predictions)
        
        if not self.available_patients:
            print("No patients found")
        else:
            gt_zero_count = len(self.available_patients) - len(patients_with_predictions)
            print(f"Found {len(self.available_patients)} patients total:")
            print(f"  - {len(patients_with_predictions)} with predictions")
            print(f"  - {gt_zero_count} GT=0 cases (images only)")
            print(f"  Range: {min(self.available_patients)}-{max(self.available_patients)}")
        
        
        
        
    def setup_viewer(self):
        """Setup clean viewer interface with professional layout"""
        # White background theme
        plt.style.use('default')
        self.fig = plt.figure(figsize=(18, 10), facecolor='white')
        
        # Patient title at top center
        self.fig.suptitle('', fontsize=16, fontweight='bold', y=0.95)
        
        # Professional grid layout with proper spacing
        self.axes = {}
        
        # Three main image panels - equal size and properly spaced
        self.axes['dwi'] = plt.subplot2grid((12, 20), (1, 0), colspan=6, rowspan=7)
        self.axes['adc'] = plt.subplot2grid((12, 20), (1, 7), colspan=6, rowspan=7) 
        self.axes['overlay'] = plt.subplot2grid((12, 20), (1, 14), colspan=6, rowspan=7)
        
        # Clean metrics panel
        self.axes['metrics'] = plt.subplot2grid((12, 20), (9, 0), colspan=20, rowspan=2)
        
        # Slice navigation slider - full width
        slice_ax = plt.subplot2grid((12, 20), (8, 2), colspan=16, rowspan=1)
        self.widgets['slice_slider'] = Slider(
            slice_ax, 'Slice', 0, 143, valinit=72, valfmt='%d'
        )
        self.widgets['slice_slider'].on_changed(self.update_slice)
        
        # Control buttons - evenly spaced at bottom
        button_width = 3
        button_height = 1
        button_y = 11
        
        # Navigation buttons
        prev_ax = plt.subplot2grid((12, 20), (button_y, 0), colspan=button_width, rowspan=button_height)
        self.widgets['prev_btn'] = Button(prev_ax, 'Previous', color='lightblue', hovercolor='skyblue')
        self.widgets['prev_btn'].on_clicked(self.prev_patient)
        
        # Patient input
        patient_input_ax = plt.subplot2grid((12, 20), (button_y, 4), colspan=2, rowspan=button_height)
        self.widgets['patient_input'] = TextBox(patient_input_ax, 'Patient:', initial='1')
        self.widgets['patient_input'].on_submit(self.goto_patient)
        
        next_ax = plt.subplot2grid((12, 20), (button_y, 7), colspan=button_width, rowspan=button_height)
        self.widgets['next_btn'] = Button(next_ax, 'Next', color='lightblue', hovercolor='skyblue')
        self.widgets['next_btn'].on_clicked(self.next_patient)
        
        # View mode buttons
        gt_ax = plt.subplot2grid((12, 20), (button_y, 11), colspan=button_width, rowspan=button_height)
        self.widgets['gt_btn'] = Button(gt_ax, 'Ground Truth', color='lightgreen', hovercolor='lightcoral')
        self.widgets['gt_btn'].on_clicked(lambda x: self.set_mode('gt'))
        
        pred_ax = plt.subplot2grid((12, 20), (button_y, 15), colspan=button_width, rowspan=button_height)
        self.widgets['pred_btn'] = Button(pred_ax, 'Prediction', color='orange', hovercolor='yellow')
        self.widgets['pred_btn'].on_clicked(lambda x: self.set_mode('pred'))
        
        overlay_ax = plt.subplot2grid((12, 20), (button_y, 18), colspan=2, rowspan=button_height)
        self.widgets['overlay_btn'] = Button(overlay_ax, 'Overlay', color='lightcoral', hovercolor='pink')
        self.widgets['overlay_btn'].on_clicked(lambda x: self.set_mode('overlay'))
        
        # Professional styling for image axes
        for ax_name, ax in self.axes.items():
            if ax_name in ['dwi', 'adc', 'overlay']:
                ax.set_facecolor('white')
                ax.set_aspect('equal')
                # Light blue professional border
                for spine in ax.spines.values():
                    spine.set_color('lightsteelblue')
                    spine.set_linewidth(2)
            elif ax_name == 'metrics':
                # Clean white metrics area
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                # No borders
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
        # Proper spacing to prevent overlaps
        plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, 
                           wspace=0.25, hspace=0.35)
        
    def load_patient_data(self):
        """Load current patient data"""
        if not self.available_patients:
            print("No patients available for viewing")
            return
            
        patient_num = self.available_patients[self.current_patient_idx]
        patient_id = f"sub-strokecase{patient_num:04d}_ses-0001"
        
        try:
            # Load modalities - aligned_multimodal_reg uses flair_registered folder
            self.current_data = {}
            
            # DWI
            dwi_path = self.aligned_dir / "dwi" / f"dwi_aligned_{patient_id}_DWI.nii.gz"
            if dwi_path.exists():
                img = nib.load(dwi_path)
                self.current_data['dwi'] = img.get_fdata()
            else:
                self.current_data['dwi'] = None
                
            # ADC
            adc_path = self.aligned_dir / "adc" / f"adc_aligned_{patient_id}_ADC.nii.gz"
            if adc_path.exists():
                img = nib.load(adc_path)
                self.current_data['adc'] = img.get_fdata()
            else:
                self.current_data['adc'] = None
                
            
            # Load ground truth
            gt_path = self.aligned_dir / "masks" / f"mask_aligned_{patient_id}_msk.nii.gz"
            if gt_path.exists():
                gt_img = nib.load(gt_path)
                self.current_data['gt'] = gt_img.get_fdata()
            else:
                self.current_data['gt'] = None
                
            # Load prediction
            pred_path = self.batch_dir / f"prediction_{patient_id}.nii.gz"
            if pred_path.exists():
                pred_img = nib.load(pred_path)
                self.current_data['pred'] = pred_img.get_fdata()
            else:
                self.current_data['pred'] = None
                
            # Get statistics with comprehensive metrics
            if patient_num in self.patients_data:
                patient_info = self.patients_data[patient_num]
                # Get volumes
                gt_volume = patient_info.get('gt_volume', 0)
                pred_volume = patient_info.get('lesion_volume', 0)
                
                # Calculate all metrics if we have the data
                gt_data = self.current_data.get('gt')
                pred_data = self.current_data.get('pred')
                
                if gt_data is not None and pred_data is not None:
                    dice = calculate_dice_score(pred_data, gt_data)
                    iou = calculate_iou_score(pred_data, gt_data)
                    sensitivity = calculate_sensitivity(pred_data, gt_data)
                    specificity = calculate_specificity(pred_data, gt_data)
                    precision = calculate_precision(pred_data, gt_data)
                else:
                    dice = patient_info['dice_score']
                    iou = sensitivity = specificity = precision = 0.0
                
                self.lesion_stats = {
                    'dice': dice,
                    'iou': iou,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'gt_volume': gt_volume,
                    'pred_volume': pred_volume
                }
            else:
                # Calculate all metrics manually if not in JSON
                gt_data = self.current_data.get('gt')
                pred_data = self.current_data.get('pred')
                
                if gt_data is not None and pred_data is not None:
                    gt_volume = int(np.sum(gt_data > 0.5))
                    pred_volume = int(np.sum(pred_data > 0.5))
                    
                    # Calculate all metrics using imported functions
                    dice = calculate_dice_score(pred_data, gt_data)
                    iou = calculate_iou_score(pred_data, gt_data)
                    sensitivity = calculate_sensitivity(pred_data, gt_data)
                    specificity = calculate_specificity(pred_data, gt_data)
                    precision = calculate_precision(pred_data, gt_data)
                    
                    self.lesion_stats = {
                        'dice': dice,
                        'iou': iou,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'precision': precision,
                        'gt_volume': gt_volume,
                        'pred_volume': pred_volume
                    }
                else:
                    self.lesion_stats = {
                        'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 
                        'specificity': 0.0, 'precision': 0.0, 
                        'gt_volume': 0, 'pred_volume': 0
                    }
                
            # Validate that we have at least some data to display
            has_data = False
            for key in ['dwi', 'adc', 'pred']:
                if self.current_data.get(key) is not None:
                    has_data = True
                    break
            
            if not has_data:
                print(f"Warning: No image data loaded for patient {patient_num}")
                print(f"Checked paths: DWI, ADC, GT, Prediction")
                return
            
            self.update_display()
            self.print_patient_info()
            
            # Update patient input box
            if hasattr(self, 'widgets') and 'patient_input' in self.widgets:
                self.widgets['patient_input'].set_val(str(patient_num))
            
        except Exception as e:
            print(f"Error loading patient {patient_num}: {e}")
            print(f"Available patients: {self.available_patients}")
            
    def print_patient_info(self):
        """Print patient information"""
        patient_num = self.available_patients[self.current_patient_idx]
        
        dice = self.lesion_stats.get('dice', 0.0)
        gt_vol = self.lesion_stats.get('gt_volume', 0)
        pred_vol = self.lesion_stats.get('pred_volume', 0)
        
        print(f"Patient {patient_num:03d} | GT: {gt_vol:,} | Pred: {pred_vol:,} | Dice: {dice:.3f}")
        
    def update_display(self):
        """Update all displays"""
        if not self.current_data:
            return
            
        slice_idx = self.current_slice
        patient_num = self.available_patients[self.current_patient_idx]
        dice = self.lesion_stats.get('dice', 0.0)
        
        # Check if this is a GT=0 case (no predictions available)
        is_gt_zero_case = patient_num not in self.patients_with_predictions
        
        # Update main title (removed dice score - now in metrics block)
        if is_gt_zero_case:
            title_parts = [f'Patient {patient_num:03d}', 'GT=0 Case (No Predictions)', f'Slice: {slice_idx}']
        else:
            title_parts = [f'Patient {patient_num:03d}', f'Slice: {slice_idx}']
            
            # Add correction indicator if using corrected data
            if hasattr(self, 'using_corrected') and self.using_corrected:
                if patient_num in self.patients_data and self.patients_data[patient_num].get('false_positive_suppressed', False):
                    title_parts.insert(-1, 'CORRECTED')
                
        title = ' | '.join(title_parts)
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # DWI - grayscale only
        ax = self.axes['dwi']
        ax.clear()
        if self.current_data['dwi'] is not None:
            data_slice = self.current_data['dwi'][:, :, slice_idx]
            ax.imshow(data_slice.T, cmap='gray', origin='lower')
        ax.set_title('DWI', fontsize=10, fontweight='bold', color='black')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # ADC - grayscale only (no color)
        ax = self.axes['adc']
        ax.clear()
        if self.current_data['adc'] is not None:
            data_slice = self.current_data['adc'][:, :, slice_idx]
            ax.imshow(data_slice.T, cmap='gray', origin='lower')
        ax.set_title('ADC', fontsize=10, fontweight='bold', color='black')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Overlay - Same background as other images with clear overlay
        ax = self.axes['overlay']
        ax.clear()
        
        # Same DWI background as other images (full opacity)
        if self.current_data['dwi'] is not None:
            dwi_slice = self.current_data['dwi'][:, :, slice_idx]
            ax.imshow(dwi_slice.T, cmap='gray', origin='lower')
        
        # Clear overlay visualization based on mode
        if self.show_mode == 'gt' and self.current_data['gt'] is not None:
            gt_slice = self.current_data['gt'][:, :, slice_idx]
            # Pure green for ground truth
            gt_masked = np.ma.masked_where(gt_slice <= 0.5, gt_slice)
            ax.imshow(gt_masked.T, cmap='Greens', alpha=0.6, origin='lower', vmin=0, vmax=1)
            
        elif self.show_mode == 'pred':
            if is_gt_zero_case:
                # For GT=0 cases, show brain image instead of prediction
                pass  # DWI background already shown above
            elif self.current_data['pred'] is not None:
                pred_slice = self.current_data['pred'][:, :, slice_idx]
                # Pure red for prediction
                pred_masked = np.ma.masked_where(pred_slice <= 0.5, pred_slice)
                ax.imshow(pred_masked.T, cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=1)
            
        elif self.show_mode == 'overlay':
            if is_gt_zero_case:
                # For GT=0 cases, show brain image instead of overlay
                pass  # DWI background already shown above
            elif self.current_data['gt'] is not None and self.current_data['pred'] is not None:
                # Green=GT, Red=Pred, Yellow=Both
                gt_slice = self.current_data['gt'][:, :, slice_idx]
                pred_slice = self.current_data['pred'][:, :, slice_idx]
                
                # Create masks
                gt_mask = gt_slice > 0.5
                pred_mask = pred_slice > 0.5
                both_mask = gt_mask & pred_mask
                
                # Create RGB overlay
                overlay_rgb = np.zeros((*gt_slice.shape, 3))
                
                # Green for GT only
                gt_only = gt_mask & ~pred_mask
                overlay_rgb[gt_only, 1] = 1.0  # Pure green
                
                # Red for Pred only  
                pred_only = pred_mask & ~gt_mask
                overlay_rgb[pred_only, 0] = 1.0  # Pure red
                
                # Yellow for both (Red + Green)
                overlay_rgb[both_mask, 0] = 1.0  # Red channel
                overlay_rgb[both_mask, 1] = 1.0  # Green channel
                
                # Show overlay
                ax.imshow(overlay_rgb.transpose(1, 0, 2), alpha=0.6, origin='lower')
        
        # Overlay title with legend
        if self.show_mode == 'overlay':
            if is_gt_zero_case:
                ax.set_title('GT=0 Case\n(Brain Image Only)', fontsize=9, fontweight='bold', color='darkred')
            else:
                ax.set_title('Overlay\nGreen=GT | Red=Pred | Yellow=Both', fontsize=9, fontweight='bold', color='black')
        elif self.show_mode == 'pred':
            if is_gt_zero_case:
                ax.set_title('GT=0 Case\n(Brain Image Only)', fontsize=9, fontweight='bold', color='darkred')
            else:
                ax.set_title('PRED', fontsize=10, fontweight='bold', color='black')
        else:
            ax.set_title(f'{self.show_mode.upper()}', fontsize=10, fontweight='bold', color='black')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Update comprehensive metrics display
        self.update_metrics_display(is_gt_zero_case)
        
        plt.draw()
        
    def update_metrics_display(self, is_gt_zero_case):
        """Update clean metrics display for presentation"""
        ax = self.axes['metrics']
        ax.clear()
        
        # Clean white background - no light blue box
        ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove all borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        if is_gt_zero_case:
            # Clean GT=0 case message
            ax.text(0.5, 0.5, "GT=0 CASE - No Ground Truth Available", 
                   ha='center', va='center', fontsize=14, fontweight='bold', 
                   color='black')
        else:
            # Clean presentation metrics - simple layout with no overlap
            dice = self.lesion_stats.get('dice', 0.0)
            iou = self.lesion_stats.get('iou', 0.0)
            sensitivity = self.lesion_stats.get('sensitivity', 0.0)
            specificity = self.lesion_stats.get('specificity', 0.0)
            precision = self.lesion_stats.get('precision', 0.0)
            
            # Clean metrics text - well spaced
            metrics_text = f"DICE: {dice:.3f}    IoU: {iou:.3f}    SENS: {sensitivity:.3f}    SPEC: {specificity:.3f}    PREC: {precision:.3f}"
            
            # Simple text display - no box background
            ax.text(0.5, 0.5, metrics_text, 
                   ha='center', va='center', fontsize=13, fontweight='bold', 
                   color='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
    def update_slice(self, val):
        """Update slice"""
        self.current_slice = int(val)
        self.update_display()
        
    def set_mode(self, mode):
        """Set display mode"""
        self.show_mode = mode
        self.update_display()
        
    def prev_patient(self, event):
        """Previous patient"""
        self.current_patient_idx = (self.current_patient_idx - 1) % len(self.available_patients)
        self.load_patient_data()
        
    def next_patient(self, event):
        """Next patient"""
        self.current_patient_idx = (self.current_patient_idx + 1) % len(self.available_patients)
        self.load_patient_data()
        
    def goto_patient(self, text):
        """Go to specific patient by number"""
        if not self.available_patients:
            print("No patients available for viewing")
            return
            
        try:
            patient_num = int(text.strip())
            if patient_num in self.available_patients:
                self.current_patient_idx = self.available_patients.index(patient_num)
                self.load_patient_data()
            else:
                print(f"Patient {patient_num} not found. Available: {min(self.available_patients)}-{max(self.available_patients)}")
                # Reset to current patient number
                current_patient = self.available_patients[self.current_patient_idx]
                self.widgets['patient_input'].set_val(str(current_patient))
        except ValueError:
            print("Invalid patient number. Please enter a number.")
            # Reset to current patient number
            current_patient = self.available_patients[self.current_patient_idx]
            self.widgets['patient_input'].set_val(str(current_patient))
        
    def show(self):
        """Display viewer"""
        print("Clinical Viewer")
        print("="*80)
        plt.show()

def launch_clinical_viewer():
    """Launch the viewer"""
    try:
        viewer = ClinicalSliceViewer()
        viewer.show()
        return viewer
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    viewer = launch_clinical_viewer()