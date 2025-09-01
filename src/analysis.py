def calculate_volumes(seg_mask_data, affine):
    """
    Calculates the volume of tumor regions in cm³.

    Args:
        seg_mask_data (np.array): The 3D numpy array of the predicted segmentation.
        affine (np.array): The affine matrix from the original NIfTI file.

    Returns:
        dict: A dictionary containing the volumes of ET, TC, and WT in cm³.
    """
    #Calculate the volume of a single voxel in mm³
    voxel_volume_mm3 = np.abs(np.linalg.det(affine[:3, :3]))

    # Convert voxel volume to cm³ (1 cm³ = 1000 mm³)
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

    # Count the number of voxels for each class
    # Class 0: Background, 1: Necrotic Core (NCR), 2: Edema (ED), 3: Enhancing Tumor (ET)
    ncr_voxels = np.sum(seg_mask_data == 1)
    ed_voxels = np.sum(seg_mask_data == 2)
    et_voxels = np.sum(seg_mask_data == 3)

    # Calculate volumes based on definitions
    et_volume = et_voxels * voxel_volume_cm3
    tc_volume = (ncr_voxels + et_voxels) * voxel_volume_cm3 # Tumor Core = NCR + ET
    wt_volume = (ncr_voxels + et_voxels + ed_voxels) * voxel_volume_cm3 # Whole Tumor = NCR + ET + ED

    return {'ET': et_volume, 'TC': tc_volume, 'WT': wt_volume}


def generate_radiology_report(tumor_volumes):
    """
    Generates a descriptive report using Gemini based on segmentation volumes.
    """
    if not GEMINI_API_KEY:
        return "Report generation is unavailable because the Gemini API key is not configured."

    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are an assistant aiding a radiologist. Based on AI-powered 3D segmentation, generate a structured, descriptive report.

    **Quantitative Analysis Data:**
    - Estimated Tumor Volumes:
      - Enhancing Tumor (ET): {tumor_volumes['ET']:.2f} cm³
      - Tumor Core (TC): {tumor_volumes['TC']:.2f} cm³
      - Whole Tumor (WT): {tumor_volumes['WT']:.2f} cm³

    **Instructions:**
    1.  Start with "Automated Brain Tumor Segmentation Report".
    2.  Create a "Summary" section interpreting the findings.
    3.  Create a "Quantitative Results" section with the volumes.
    4.  Add a "Disclaimer" that this is AI-generated and requires professional verification.
    5.  Maintain a professional, clinical tone. Do NOT provide a diagnosis.
    """
    response = model.generate_content(prompt)
    return response.text