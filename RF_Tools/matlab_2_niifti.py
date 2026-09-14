import os
import numpy as np
import scipy.io as spio
import nibabel as nib

def convert_mat_to_nifti(mat_path, output_nii_path):
    print(f"Lecture du fichier MATLAB : {mat_path}")
    
    # 1. Charger le fichier .mat
    mat_data = spio.loadmat(mat_path)
    
    # 2. Identifier les variables clés à l'intérieur
    # (Parfois les structures MATLAB sont imbriquées, on liste les clés pour comprendre)
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"Variables trouvées dans le fichier : {keys}")
    
    # Hypothèse standard : Votre matrice d'image est stockée dans une variable.
    # Remplacer 'vol', 'img' ou le nom de votre structure si nécessaire.
    # Si c'est une structure imbriquée complexe (ex: de SPM), on cible le champ de données.
    img_data = None
    affine = np.eye(4)  # Matrice affine par défaut (identité) si absente du .mat
    
    # Exemple d'extraction automatique si une variable ressemble à une image 3D/4D
    for key in keys:
        var = mat_data[key]
        # Si la variable est une matrice à 3 ou 4 dimensions
        if isinstance(var, np.ndarray) and var.ndim in [3, 4]:
            img_data = var
            print(f"Image trouvée dans la variable '{key}' de forme (shape) : {img_data.shape}")
            break
        # Si la variable est une structure (array de type void)
        elif isinstance(var, np.ndarray) and var.dtype.names is not None:
            print(f"Analyse de la structure complexe : {key}")
            # Tente de trouver les champs typiques d'une structure NIfTI MATLAB (comme dans SPM)
            # Champs courants : 'dat', 'img', 'vol', 'mat', 'affine'
            names = var.dtype.names
            if 'img' in names:
                img_data = var['img'][0, 0]
            elif 'dat' in names:
                img_data = var['dat'][0, 0]
                
            if 'mat' in names:
                affine = var['mat'][0, 0]
            elif 'affine' in names:
                affine = var['affine'][0, 0]
                
    if img_data is None:
        raise ValueError("Impossible de trouver une matrice d'image 3D/4D valide dans le fichier .mat. "
                         "Veuillez vérifier manuellement le nom de la variable.")

    # 3. Correction d'orientation (Ajustement MATLAB vs Python)
    # MATLAB est en Fortran-order (indexé par colonnes) tandis que Python est en C-order.
    # Si votre image apparaît "tournée" ou inversée au final, décommentez la ligne suivante :
    # img_data = np.transpose(img_data, (2, 1, 0)) 

    # Assurez-vous que l'affine fait bien 4x4
    if affine.shape != (4, 4):
        print("Avertissement : La matrice affine trouvée n'est pas au format 4x4. Utilisation de la matrice identité.")
        affine = np.eye(4)

    # 4. Création de l'objet NIfTI
    print("Création de l'image NIfTI...")
    nifti_img = nib.Nifti1Image(img_data, affine=affine)
    
    # Optionnel : Forcer le type de données pour éviter la troncature/perte de précision
    # (ex: float32 ou int16 selon vos données d'origine)
    nifti_img.set_data_dtype(img_data.dtype)

    # 5. Sauvegarde sur le disque
    nib.save(nifti_img, output_nii_path)
    print(f"Conversion réussie ! Fichier enregistré sous : {output_nii_path}")

def convert_all_mat_to_nifti(mat_path, output_dir="export_nifti"):
    """
    Parcourt un fichier .mat et exporte chaque matrice d'image (2D, 3D ou 4D) 
    dans un fichier NIfTI distinct, en ignorant proprement les structures de configuration.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Lecture du fichier MATLAB : {mat_path}")
    mat_data = spio.loadmat(mat_path)
    
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    print(f"Variables détectées dans le fichier : {keys}")
    print("-" * 60)
    
    affine = np.eye(4)  # Matrice affine par défaut
    
    for key in keys:
        data = mat_data[key]
        
        # Sécurité 1 : Est-ce un array numpy ?
        if not isinstance(data, np.ndarray):
            print(f"Saut de '{key}' : Pas une matrice de données.")
            continue
            
        # Sécurité 2 : Est-ce une structure MATLAB (comme 'opt') ?
        if data.dtype.names is not None:
            print(f"Saut de '{key}' : Ignoré (Structure MATLAB détectée avec les champs : {list(data.dtype.names)})")
            continue
            
        dimensions = data.ndim
        
        # Sécurité 3 : Dimension ou taille insuffisante (ex: scalaires, vecteurs)
        if dimensions < 2 or (dimensions == 2 and (data.shape[0] <= 1 or data.shape[1] <= 1)):
            print(f"Saut de '{key}' : Ignoré (Format ou taille non compatible image : {data.shape})")
            continue
            
        print(f"Traitement de la variable '{key}' | Forme d'origine : {data.shape}")
        
        # Ajustement des dimensions pour le standard NIfTI
        if dimensions == 2:
            # Image 2D (ex: 192x256) -> devient 3D (192x256x1)
            img_to_save = data[:, :, np.newaxis]
            print(f"  -> Format 2D détecté. Ajustement pour le NIfTI : {img_to_save.shape}")
        elif dimensions == 4 and data.shape[2] == 1:
            # Cas de votre variable 'img' qui fait (192, 256, 1, 24)
            # C'est déjà parfait pour le NIfTI (X, Y, Z=1, Temps/Echos=24)
            img_to_save = data
            print(f"  -> Format 4D détecté (Série temporelle/multi-écho).")
        else:
            img_to_save = data
            print(f"  -> Format {dimensions}D détecté.")
            
        # Création et sauvegarde du NIfTI
        try:
            nifti_img = nib.Nifti1Image(img_to_save, affine=affine)
            nifti_img.set_data_dtype(img_to_save.dtype)
            
            base_name = os.path.splitext(os.path.basename(mat_path))[0]
            file_name = f"{base_name}_{key}.nii.gz"
            full_output_path = os.path.join(output_dir, file_name)
            
            nib.save(nifti_img, full_output_path)
            print(f"  [SUCCÈS] Enregistré sous : {full_output_path}\n")
        except Exception as e:
            print(f"  [ERREUR] Impossible de convertir '{key}' : {e}\n")

# --- Exécution ---
if __name__ == "__main__":
    # N'hésitez pas à adapter les chemins absolus si nécessaire
    mat_file = "C:/d/pulseConversion/test110626.mat"
    output_directory = "C:/d/pulseConversion/"
    convert_all_mat_to_nifti(mat_file, output_dir=output_directory)
