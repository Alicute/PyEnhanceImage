# 测试数据说明

## 测试DICOM文件

如果您没有现成的DICOM文件用于测试，可以通过以下方式获取：

### 1. 在线DICOM样本库
- [DICOM Sample Images](https://www.rubomedical.com/dicom_files/dicom_sample_images.htm)
- [OsiriX Dicom Image Library](https://www.osirix-viewer.com/resources/dicom-image-library/)
- [CT Medical Images](https://www.cancerimagingarchive.net/)

### 2. 创建测试DICOM文件
如果需要创建简单的测试DICOM文件，可以使用以下Python脚本：

```python
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset

# 创建一个简单的测试DICOM文件
def create_test_dicom(filename, size=(512, 512)):
    # 创建像素数据（模拟16位医学图像）
    pixel_array = np.random.randint(0, 65535, size, dtype=np.uint16)
    
    # 添加一些模拟的结构
    center_x, center_y = size[0] // 2, size[1] // 2
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (min(size) // 4)**2
    pixel_array[mask] = pixel_array[mask] + 10000
    
    # 创建DICOM数据集
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9.0'
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    # 添加必需的DICOM标签
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.StudyDescription = "Test Study"
    ds.SeriesDescription = "Test Series"
    ds.ImageComments = "Test image for PyEnhanceImage"
    
    # 设置图像信息
    ds.Rows = size[0]
    ds.Columns = size[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.tobytes()
    
    # 保存文件
    ds.save_as(filename)
    print(f"Created test DICOM file: {filename}")

# 使用示例
if __name__ == "__main__":
    create_test_dicom("test_image.dcm")
```

### 3. 使用其他格式转换
如果您有其他格式的医学图像，可以使用工具转换为DICOM格式：
- [ImageJ](https://imagej.nih.gov/ij/) - 支持多种医学图像格式
- [3D Slicer](https://www.slicer.org/) - 专业的医学图像处理软件
- [Horos](https://horosproject.org/) - 免费的DICOM查看器

## 使用说明

1. 将DICOM文件放在项目目录或任何其他位置
2. 点击应用程序中的"加载DICOM"按钮
3. 选择您的DICOM文件
4. 开始使用各种图像增强功能

## 注意事项

- 确保DICOM文件是有效的医学图像文件
- 支持单通道16位灰度图像
- 如果加载失败，请检查文件格式和完整性