#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态文档加载器演示

本演示展示如何从PDF、Word、Markdown文件中提取图片，
模拟上传到腾讯云COS，并将图片引用替换为云存储URL。

功能特性：
1. 支持PDF、Word、Markdown文件的图片提取
2. 模拟图片上传到腾讯云COS
3. 将原文档中的图片引用替换为云存储URL
4. 完整的错误处理和日志记录

依赖安装：
pip install pypdf python-docx pillow requests
"""

import os
import re
import io
import base64
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    print("请安装PyMuPDF: pip install PyMuPDF")
    fitz = None

try:
    from docx import Document
except ImportError:
    print("请安装python-docx: pip install python-docx")
    Document = None

try:
    from qcloud_cos import CosConfig
    from qcloud_cos import CosS3Client
except ImportError:
    print("请安装腾讯云COS SDK: pip install cos-python-sdk-v5")
    CosConfig = None
    CosS3Client = None

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 导入配置
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.config import load_cos_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalDocumentLoader:
    """
    多模态文档加载器
    支持从PDF、Word、Markdown等文档中提取图片，并上传到腾讯云COS
    """
    
    def __init__(self, cos_bucket: str = None, cos_region: str = None):
        """
        初始化多模态文档加载器
        
        Args:
            cos_bucket: COS存储桶名称（可选，从环境变量读取）
            cos_region: COS区域（可选，从环境变量读取）
        """
        # 加载COS配置
        try:
            cos_config = load_cos_config()
            self.cos_secret_id = cos_config['secret_id']
            self.cos_secret_key = cos_config['secret_key']
            self.cos_region = cos_config['region']
            self.cos_bucket = cos_config['bucket']
            self.cos_domain = cos_config.get('domain')
            
            print(f"🔧 COS配置加载成功: {self.cos_bucket} ({self.cos_region})")
            
            # 初始化COS客户端
            if CosConfig and CosS3Client:
                config = CosConfig(Region=self.cos_region, 
                                 SecretId=self.cos_secret_id, 
                                 SecretKey=self.cos_secret_key)
                self.cos_client = CosS3Client(config)
                self.cos_enabled = True
            else:
                self.cos_client = None
                self.cos_enabled = False
                print("⚠️ COS SDK未安装，将使用模拟上传模式")
                
        except Exception as e:
            print(f"⚠️ COS配置加载失败: {e}")
            print("将使用模拟上传模式，请检查.env文件中的COS配置")
            self.cos_enabled = False
            self.cos_client = None
            # 从环境变量中读取COS配置
            self.cos_bucket = cos_bucket or os.getenv('COS_BUCKET')
            self.cos_region = cos_region or os.getenv('COS_REGION')
            self.cos_domain = os.getenv('COS_DOMAIN')
        
        # 设置基础URL
        if hasattr(self, 'cos_domain') and self.cos_domain:
            # 如果cos_domain已经包含协议，直接使用；否则添加https://
            if self.cos_domain.startswith(('http://', 'https://')):
                self.base_url = self.cos_domain
            else:
                self.base_url = f"https://{self.cos_domain}"
        else:
            self.base_url = f"https://{self.cos_bucket}.cos.{self.cos_region}.myqcloud.com"
        
        # 创建临时图片存储目录
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"多模态文档加载器初始化完成，COS配置: {self.base_url}")
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF文件中提取图片
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            图片信息列表，包含图片数据、页码、位置等
        """
        images = []
        
        try:
            # 使用PyMuPDF打开PDF
            doc = fitz.open(pdf_path)
            logger.info(f"开始从PDF提取图片: {pdf_path}，共{len(doc)}页")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    # 获取图片对象
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # 转换为PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_pil = Image.open(io.BytesIO(img_data))
                        
                        # 生成图片哈希作为文件名
                        img_hash = hashlib.md5(img_data).hexdigest()
                        
                        images.append({
                            'data': img_data,
                            'format': 'png',
                            'page': page_num + 1,
                            'index': img_index,
                            'hash': img_hash,
                            'size': img_pil.size,
                            'filename': f"pdf_page{page_num+1}_img{img_index}_{img_hash[:8]}.png"
                        })
                        
                        logger.info(f"提取图片: 页面{page_num+1}, 索引{img_index}, 尺寸{img_pil.size}")
                    
                    pix = None  # 释放内存
            
            doc.close()
            logger.info(f"PDF图片提取完成，共提取{len(images)}张图片")
            
        except Exception as e:
            logger.error(f"PDF图片提取失败: {e}")
            
        return images
    
    def extract_images_from_word(self, word_path: str) -> List[Dict]:
        """
        从Word文档中提取图片
        
        Args:
            word_path: Word文档路径
            
        Returns:
            图片信息列表
        """
        images = []
        
        try:
            logger.info(f"开始从Word文档提取图片: {word_path}")
            
            # Word文档实际上是一个ZIP文件
            with zipfile.ZipFile(word_path, 'r') as docx:
                # 查找媒体文件
                media_files = [f for f in docx.namelist() if f.startswith('word/media/')]
                
                for media_file in media_files:
                    # 读取图片数据
                    img_data = docx.read(media_file)
                    
                    # 获取文件扩展名
                    file_ext = Path(media_file).suffix.lower()
                    if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                        # 生成图片哈希
                        img_hash = hashlib.md5(img_data).hexdigest()
                        
                        # 获取图片尺寸
                        try:
                            img_pil = Image.open(io.BytesIO(img_data))
                            size = img_pil.size
                        except:
                            size = (0, 0)
                        
                        images.append({
                            'data': img_data,
                            'format': file_ext[1:],  # 去掉点号
                            'original_path': media_file,
                            'hash': img_hash,
                            'size': size,
                            'filename': f"word_{Path(media_file).stem}_{img_hash[:8]}{file_ext}"
                        })
                        
                        logger.info(f"提取图片: {media_file}, 尺寸{size}")
            
            logger.info(f"Word图片提取完成，共提取{len(images)}张图片")
            
        except Exception as e:
            logger.error(f"Word图片提取失败: {e}")
            
        return images
    
    def extract_images_from_markdown(self, md_path: str) -> List[Dict]:
        """
        从Markdown文件中提取图片引用
        
        Args:
            md_path: Markdown文件路径
            
        Returns:
            图片信息列表
        """
        images = []
        
        try:
            logger.info(f"开始从Markdown文件提取图片引用: {md_path}")
            
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 匹配Markdown图片语法: ![alt](path)
            img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
            matches = re.finditer(img_pattern, content)
            
            md_dir = Path(md_path).parent
            
            for match in matches:
                alt_text = match.group(1)
                img_path = match.group(2)
                
                # 如果是相对路径，转换为绝对路径
                if not img_path.startswith(('http://', 'https://')):
                    full_img_path = md_dir / img_path
                    
                    if full_img_path.exists():
                        # 读取本地图片文件
                        with open(full_img_path, 'rb') as img_file:
                            img_data = img_file.read()
                        
                        # 获取图片格式
                        file_ext = full_img_path.suffix.lower()
                        
                        # 生成图片哈希
                        img_hash = hashlib.md5(img_data).hexdigest()
                        
                        # 获取图片尺寸
                        try:
                            img_pil = Image.open(io.BytesIO(img_data))
                            size = img_pil.size
                        except:
                            size = (0, 0)
                        
                        images.append({
                            'data': img_data,
                            'format': file_ext[1:] if file_ext else 'png',
                            'alt_text': alt_text,
                            'original_path': img_path,
                            'full_path': str(full_img_path),
                            'hash': img_hash,
                            'size': size,
                            'filename': f"md_{full_img_path.stem}_{img_hash[:8]}{file_ext}",
                            'markdown_syntax': match.group(0)
                        })
                        
                        logger.info(f"提取图片: {img_path}, 尺寸{size}")
                    else:
                        logger.warning(f"图片文件不存在: {full_img_path}")
            
            logger.info(f"Markdown图片提取完成，共提取{len(images)}张图片")
            
        except Exception as e:
            logger.error(f"Markdown图片提取失败: {e}")
            
        return images
    
    def upload_to_cos(self, image_data: bytes, filename: str) -> str:
        """
        上传图片到腾讯云COS
        
        Args:
            image_data: 图片二进制数据
            filename: 文件名
            
        Returns:
            COS URL或模拟URL
        """
        try:
            if self.cos_enabled and self.cos_client:
                # 真实上传到COS
                object_key = f"images/{filename}"
                
                # 上传文件
                response = self.cos_client.put_object(
                    Bucket=self.cos_bucket,
                    Body=image_data,
                    Key=object_key,
                    ContentType=self._get_content_type(filename)
                )
                
                # 生成访问URL
                cos_url = f"{self.base_url}/{object_key}"
                
                logger.info(f"✅ COS上传成功: {filename} -> {cos_url}")
                return cos_url
                
            else:
                # 模拟上传模式
                return self._simulate_cos_upload(image_data, filename)
                
        except Exception as e:
            logger.error(f"❌ COS上传失败: {e}")
            # 失败时回退到模拟模式
            return self._simulate_cos_upload(image_data, filename)
    
    def _simulate_cos_upload(self, image_data: bytes, filename: str) -> str:
        """
        模拟上传图片到腾讯云COS
        
        Args:
            image_data: 图片二进制数据
            filename: 文件名
            
        Returns:
            模拟的COS URL
        """
        try:
            # 保存到临时目录（模拟上传）
            temp_file = self.temp_dir / filename
            with open(temp_file, 'wb') as f:
                f.write(image_data)
            
            # 生成模拟的COS URL
            cos_url = f"{self.base_url}/images/{filename}"
            
            logger.info(f"🔄 模拟上传: {filename} -> {cos_url}")
            return cos_url
            
        except Exception as e:
            logger.error(f"❌ 模拟上传失败: {e}")
            return ""
    
    def _get_content_type(self, filename: str) -> str:
        """
        根据文件扩展名获取Content-Type
        
        Args:
            filename: 文件名
            
        Returns:
            Content-Type字符串
        """
        ext = Path(filename).suffix.lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def process_document(self, file_path: str) -> Dict:
        """
        处理文档，提取图片并替换为COS URL
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            处理结果，包含原始内容、处理后内容、图片信息等
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        logger.info(f"开始处理文档: {file_path}")
        
        result = {
            'file_path': str(file_path),
            'file_type': file_ext,
            'images': [],
            'cos_urls': [],
            'original_content': '',
            'processed_content': '',
            'success': False
        }
        
        try:
            # 根据文件类型提取图片
            if file_ext == '.pdf':
                images = self.extract_images_from_pdf(str(file_path))
            elif file_ext in ['.docx', '.doc']:
                images = self.extract_images_from_word(str(file_path))
            elif file_ext in ['.md', '.markdown']:
                images = self.extract_images_from_markdown(str(file_path))
                # 读取原始Markdown内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    result['original_content'] = f.read()
            else:
                logger.warning(f"不支持的文件类型: {file_ext}")
                return result
            
            result['images'] = images
            
            # 上传图片到COS并获取URL
            cos_urls = []
            for img in images:
                cos_url = self.upload_to_cos(img['data'], img['filename'])
                if cos_url:
                    cos_urls.append({
                        'filename': img['filename'],
                        'cos_url': cos_url,
                        'original_info': img
                    })
            
            result['cos_urls'] = cos_urls
            
            # 如果是Markdown文件，替换图片引用
            if file_ext in ['.md', '.markdown'] and result['original_content']:
                processed_content = result['original_content']
                
                for i, img in enumerate(images):
                    if i < len(cos_urls):
                        # 替换Markdown图片语法
                        old_syntax = img['markdown_syntax']
                        new_syntax = f"![{img['alt_text']}]({cos_urls[i]['cos_url']})"
                        processed_content = processed_content.replace(old_syntax, new_syntax)
                
                result['processed_content'] = processed_content
            
            result['success'] = True
            logger.info(f"文档处理完成: 提取{len(images)}张图片，上传{len(cos_urls)}个URL")
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            result['error'] = str(e)
        
        return result
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("临时文件清理完成")
        except Exception as e:
            logger.error(f"临时文件清理失败: {e}")

def demo_multimodal_processing():
    """演示多模态文档处理"""
    print("=" * 60)
    print("多模态文档加载器演示")
    print("=" * 60)
    
    # 初始化加载器
    loader = MultimodalDocumentLoader(
        cos_bucket="langchain-demo",
        cos_region="ap-shanghai"
    )
    
    # 测试文件路径
    test_files = [
        "docs/struct.pdf",  # PDF文件
        "tutorials/14_multimodal_rag/test_markdown.md",  # 测试Markdown文件
        # "sample.docx",  # Word文件（如果有的话）
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n处理文件: {file_path}")
            print("-" * 40)
            
            # 处理文档
            result = loader.process_document(file_path)
            
            if result['success']:
                print(f"✅ 处理成功")
                print(f"📄 文件类型: {result['file_type']}")
                print(f"🖼️  提取图片: {len(result['images'])}张")
                print(f"☁️  上传COS: {len(result['cos_urls'])}个URL")
                
                # 显示图片信息
                for i, img in enumerate(result['images'][:3]):  # 只显示前3张
                    print(f"   图片{i+1}: {img['filename']} ({img['size'][0]}x{img['size'][1]})")
                
                # 显示COS URL
                for i, url_info in enumerate(result['cos_urls'][:3]):  # 只显示前3个
                    print(f"   URL{i+1}: {url_info['cos_url']}")
                
                # 如果是Markdown，显示内容替换效果
                if result['processed_content']:
                    print("\n📝 Markdown内容替换示例:")
                    
                    # 显示所有被替换的图片引用
                    replacement_found = False
                    for i, img in enumerate(result['images']):
                        if i < len(result['cos_urls']):
                            original_syntax = img['markdown_syntax']
                            new_syntax = f"![{img['alt_text']}]({result['cos_urls'][i]['cos_url']})"
                            print(f"   原始: {original_syntax}")
                            print(f"   替换: {new_syntax}")
                            replacement_found = True
                    
                    if not replacement_found:
                        print("   (没有图片被替换)")
            else:
                print(f"❌ 处理失败: {result.get('error', '未知错误')}")
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    # 清理临时文件
    loader.cleanup_temp_files()
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

if __name__ == "__main__":
    demo_multimodal_processing()