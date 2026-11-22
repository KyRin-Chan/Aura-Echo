import logging
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

# Import the ModelManager and TensorRTManager
from downloader.ModelManager import ModelManager
from downloader.TensorRTManager import TensorRTManager

logger = logging.getLogger(__name__)

class MMVC_Rest_PretrainDownloader:
    def __init__(self, voiceChangerManager=None):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/download_pretrain", self.download_pretrain, methods=["POST"])
        self.router.add_api_route("/delete_pretrain", self.delete_pretrain, methods=["POST"])
        self.router.add_api_route("/tensorrt/status", self.get_tensorrt_status, methods=["GET"])
        self.router.add_api_route("/tensorrt/download", self.download_tensorrt, methods=["POST"])
        self.router.add_api_route("/tensorrt/cancel", self.cancel_tensorrt_download, methods=["POST"])
        self.router.add_api_route("/tensorrt/delete", self.delete_tensorrt, methods=["POST"])

    async def download_pretrain(self, model_key: str = Form(...)):
        """
        Download a pretrained model by its key.
        
        Args:
            model_key: Key of the model (e.g., 'crepe_onnx_full', 'rmvpe_onnx')
            
        Returns:
            JSON response with status and message
        """
        try:
            result = await ModelManager.download_model(model_key)
            return JSONResponse(status_code=200, content=result)
            
        except ValueError as ve:
            status_code = 400 if "mandatory" in str(ve) else 404
            raise HTTPException(status_code=status_code, detail=str(ve))
        except Exception as e:
            logger.exception(f"Error downloading model {model_key}")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_pretrain(self, model_key: str = Form(...)):
        """
        Delete a downloaded pretrained model.
        
        Args:
            model_key: Key of the model (e.g., 'crepe_onnx_full', 'rmvpe_onnx')
            
        Returns:
            JSON response with status and message
        """
        try:
            result = ModelManager.delete_model(model_key)
            return JSONResponse(status_code=200, content=result)
            
        except ValueError as ve:
            status_code = 400 if "mandatory" in str(ve) else 404
            raise HTTPException(status_code=status_code, detail=str(ve))
        except Exception as e:
            logger.exception(f"Error deleting model {model_key}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_tensorrt_status(self):
        try:
            status = TensorRTManager.get_instance().get_status()
            return JSONResponse(status_code=200, content=status)
        except Exception as e:
            logger.exception("Error getting TensorRT status")
            raise HTTPException(status_code=500, detail=str(e))

    async def download_tensorrt(self):
        try:
            status = await TensorRTManager.get_instance().start_download()
            return JSONResponse(status_code=200, content=status)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re_err:
            raise HTTPException(status_code=409, detail=str(re_err))
        except Exception as e:
            logger.exception("Error starting TensorRT download")
            raise HTTPException(status_code=500, detail=str(e))

    async def cancel_tensorrt_download(self):
        try:
            status = await TensorRTManager.get_instance().cancel_download()
            return JSONResponse(status_code=200, content=status)
        except Exception as e:
            logger.exception("Error cancelling TensorRT download")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_tensorrt(self):
        try:
            result = TensorRTManager.get_instance().delete_runtime(self.voiceChangerManager)
            return JSONResponse(status_code=200, content=result)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.exception("Error deleting TensorRT runtime")
            raise HTTPException(status_code=500, detail=str(e))