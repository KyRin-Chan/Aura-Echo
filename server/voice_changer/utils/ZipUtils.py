import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Set, List, Dict, Any
import logging
import re
import urllib.parse

logger = logging.getLogger(__name__)

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename to ensure it only contains ASCII alphanumeric characters,
    underscores, hyphens, and dots.
    If the name becomes empty, uses a default based on extension.
    """
    if not filename:
        return ""
    try:
        filename = urllib.parse.unquote(filename)
    except Exception:
        pass
        
    base, ext = os.path.splitext(filename)
    
    # Replace spaces with underscores
    base = base.replace(' ', '_')
    # Keep only alphanumeric, hyphens, underscores, dots
    sanitized_base = re.sub(r'[^a-zA-Z0-9_\-.]', '', base)
    
    # If the sanitized base name is empty, fallback to a default
    if not sanitized_base:
        if ext.lower() in ('.index',):
            sanitized_base = "index"
        elif ext.lower() in ('.onnx',):
            sanitized_base = "model"
        elif ext.lower() in ('.pth', '.pt', '.safetensors'):
            sanitized_base = "model"
        else:
            sanitized_base = "file"
            
    sanitized_ext = re.sub(r'[^a-zA-Z0-9.]', '', ext)
    
    return f"{sanitized_base}{sanitized_ext}"

class FileUtils:
    """Utility class for file operations including ZIP handling."""
    
    @staticmethod
    def extract_zip(
        zip_path: str, 
        extract_to: str, 
        allowed_model_extensions: Optional[Set[str]] = None,
        allowed_index_extensions: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract zip file and return information about extracted files.
        
        Args:
            zip_path: Path to the ZIP file
            extract_to: Directory to extract files to
            allowed_model_extensions: Set of allowed model file extensions
            allowed_index_extensions: Set of allowed index file extensions
            
        Returns:
            Dict containing information about extracted files
        """
        model_exts = allowed_model_extensions or {'.pth', '.pt', '.safetensors', '.onnx'}
        index_exts = allowed_index_extensions or {'.index'}
        
        result = {
            'success': False,
            'model_file': None,
            'index_file': None,
            'files': [],
            'error': None
        }
        
        temp_dir = None
        
        try:
            if not os.path.exists(zip_path) or not zipfile.is_zipfile(zip_path):
                result['error'] = "Invalid or missing ZIP file"
                return result
                
            temp_dir = os.path.join(os.path.dirname(extract_to), f"temp_{os.urandom(8).hex()}")
            os.makedirs(temp_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_contents = zip_ref.namelist()
                
                # Find all model and index files recursively
                model_files = []
                index_files = []
                
                for file in zip_contents:
                    # Skip directories
                    if file.endswith('/') or file.endswith('\\'):
                        continue
                        
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in model_exts):
                        model_files.append(file)
                    elif any(file_lower.endswith(ext) for ext in index_exts):
                        index_files.append(file)
                
                if not model_files:
                    result['error'] = "No valid model file found in ZIP archive"
                    return result
                
                # If multiple model files found, prefer files in root directory or with shorter paths
                if len(model_files) > 1:
                    # Sort by path depth (shallow first) and then alphabetically
                    model_files.sort(key=lambda x: (x.count('/') + x.count('\\'), x.lower()))
                    # Keep only the first (best) model file
                    model_files = [model_files[0]]
                
                # Get all files to extract (model file + all index files)
                files_to_extract = model_files + index_files
                
                # Create target directory if it doesn't exist
                os.makedirs(extract_to, exist_ok=True)
                
                # Extract files
                extracted_files = []
                for file in files_to_extract:
                    try:
                        # Split by slash to sanitize each path component
                        parts = re.split(r'[/\\]', file)
                        sanitized_parts = [sanitize_filename(p) if p else '' for p in parts]
                        sanitized_file = '/'.join(p for p in sanitized_parts if p)

                        zip_ref.extract(file, temp_dir)
                        src_path = os.path.join(temp_dir, file)
                        dest_path = os.path.join(extract_to, sanitized_file)
                        
                        # Create subdirectories if needed
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        # Move file to final destination
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.move(src_path, dest_path)
                        
                        # Update result with file info
                        file_lower = sanitized_file.lower()
                        if any(file_lower.endswith(ext) for ext in model_exts):
                            # Store relative path from ZIP root
                            result['model_file'] = sanitized_file
                            logger.info(f"Found model file: {sanitized_file}")
                        elif any(file_lower.endswith(ext) for ext in index_exts):
                            # Only update index file if not already set or if this one is in a more specific path
                            if result['index_file'] is None or sanitized_file.count('/') + sanitized_file.count('\\') < result['index_file'].count('/') + result['index_file'].count('\\'):
                                result['index_file'] = sanitized_file
                                logger.info(f"Found index file: {sanitized_file}")
                        
                        extracted_files.append(sanitized_file)
                        logger.debug(f"Extracted: {sanitized_file}")
                        
                        # If this is the model file, ensure it's in the root of the extraction directory
                        if any(file_lower.endswith(ext) for ext in model_exts) and ('/' in sanitized_file or '\\' in sanitized_file):
                            # Move the model file to the root of the extraction directory
                            base_name = os.path.basename(sanitized_file)
                            sanitized_base_name = sanitize_filename(base_name)
                            root_dest = os.path.join(extract_to, sanitized_base_name)
                            if os.path.exists(root_dest):
                                os.remove(root_dest)
                            shutil.move(dest_path, root_dest)
                            result['model_file'] = sanitized_base_name
                            logger.info(f"Moved model file to root: {sanitized_base_name}")
                    except Exception as e:
                        logger.warning(f"Failed to extract {file}: {str(e)}")
                
                if not result['model_file']:
                    result['error'] = "No valid model file found in ZIP archive"
                    return result
                
                result['success'] = True
                result['files'] = extracted_files
                
        except zipfile.BadZipFile:
            result['error'] = "Invalid or corrupted ZIP file"
        except Exception as e:
            result['error'] = f"Failed to extract ZIP file: {str(e)}"
            logger.exception("Error extracting ZIP file")
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return result
    
    @staticmethod
    def find_first_file(directory: str, extensions: Set[str]) -> Optional[str]:
        """Find the first file in directory with one of the given extensions."""
        if not os.path.isdir(directory):
            return None
            
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    return os.path.join(root, file)
        return None
    
    @staticmethod
    def safe_remove(path: str) -> bool:
        """Safely remove a file or directory."""
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            return True
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {str(e)}")
            return False
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """Ensure a directory exists, creating it if necessary."""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {str(e)}")
            return False
