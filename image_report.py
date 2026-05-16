# crack_analysis_report.py
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import os
import json
from skimage.morphology import skeletonize

# ===== 中文显示配置 =====
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False
# =======================


class CrackAnalyzer:
    def __init__(self, model_path, pixel_to_mm_ratio=None):
        self.model = YOLO(model_path)
        self.ratio = pixel_to_mm_ratio

    def analyze_single_crack(self, mask, original_shape):
        """分析单个裂纹的详细几何指标"""
        if mask is None or mask.sum() == 0:
            return None

        if mask.shape != original_shape:
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        mask_binary = (mask * 255).astype(np.uint8)

        # 裂纹面积
        area_pixel = np.sum(mask_binary > 0)

        # 裂纹周长
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter_pixel = sum(cv2.arcLength(c, True) for c in contours)

        # 最大宽度
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        max_width_pixel = np.max(dist_transform) * 2

        # 裂纹长度（骨架总像素）
        skeleton = self._skeletonize(mask_binary)
        longest_length_pixel = float(np.sum(skeleton > 0))

        # 严重程度评估
        severity = self._assess_severity(area_pixel, max_width_pixel, longest_length_pixel)

        result = {
            'area_pixel': int(area_pixel),
            'perimeter_pixel': int(perimeter_pixel),
            'max_width_pixel': float(max_width_pixel),
            'longest_length_pixel': float(longest_length_pixel),
            'severity': severity['grade'],
            'severity_score': severity['score'],
            'suggestion': severity['suggestion']
        }

        if self.ratio:
            result['area_mm2'] = round(area_pixel * (self.ratio ** 2), 2)
            result['max_width_mm'] = round(max_width_pixel * self.ratio, 2)
            result['longest_length_mm'] = round(longest_length_pixel * self.ratio, 2)

        return result

    def _skeletonize(self, mask):
        """骨架化"""
        try:
            # skeletonize 已在文件顶部导入
            skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        except ImportError:
            skeleton = self._simple_skeletonize(mask)
        except Exception as e:
            print(f"骨架化失败: {e}")
            skeleton = np.zeros_like(mask)
        return skeleton

    @staticmethod
    def _simple_skeletonize(mask):
        """简单的骨架化"""
        skeleton = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        result = np.zeros_like(mask)
        while True:
            eroded = cv2.erode(skeleton, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(skeleton, temp)
            result = cv2.bitwise_or(result, temp)
            skeleton = eroded
            if cv2.countNonZero(skeleton) == 0:
                break
        return result

    @staticmethod
    def _assess_severity(area, max_width, length):
        """评估裂纹严重程度"""
        score = 0
        if area < 1000:
            score += 10
        elif area < 5000:
            score += 20
        elif area < 20000:
            score += 30
        else:
            score += 40

        if max_width < 5:
            score += 10
        elif max_width < 15:
            score += 20
        else:
            score += 30

        if length < 100:
            score += 10
        elif length < 300:
            score += 20
        else:
            score += 30

        if score < 30:
            grade = "轻微"
            suggestion = "建议观察"
        elif score < 60:
            grade = "中度"
            suggestion = "建议维修"
        else:
            grade = "严重"
            suggestion = "立即处理"

        return {'grade': grade, 'score': score, 'suggestion': suggestion}

    def generate_report(self, img_path, save_path=None, show=True):
        """生成裂纹分析报告"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return None, None

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 推理
        pred_results = self.model(img_rgb)
        result = pred_results[0]

        # 分析裂纹
        crack_analyses = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                analysis = self.analyze_single_crack(mask, (h, w))
                if analysis:
                    analysis['id'] = i + 1
                    crack_analyses.append(analysis)

        total_area = 0
        max_width = 0
        total_length = 0

        # 创建简洁的报告图（2x2布局）
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('裂纹检测分析报告', fontsize=18, fontweight='bold', y=0.98)

        # 图1：原图
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')

        # 图2：检测结果
        axes[0, 1].imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('裂纹检测结果', fontsize=12)
        axes[0, 1].axis('off')

        # 图3：裂纹掩码热力图
        if result.masks is not None:
            combined_mask = np.zeros((h, w))
            for mask in result.masks.data.cpu().numpy():
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                combined_mask += mask
            combined_mask = np.clip(combined_mask, 0, 1)
            axes[1, 0].imshow(combined_mask, cmap='hot')
            axes[1, 0].set_title('裂纹热力图', fontsize=12)
        else:
            axes[1, 0].text(0.5, 0.5, '未检测到裂纹', ha='center', va='center')
            axes[1, 0].set_title('裂纹热力图', fontsize=12)
        axes[1, 0].axis('off')

        # 图4：分析结果表格
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')

        if crack_analyses:
            # 汇总信息
            total_area = sum(c['area_pixel'] for c in crack_analyses)
            max_width = max(c['max_width_pixel'] for c in crack_analyses)
            total_length = sum(c['longest_length_pixel'] for c in crack_analyses)

            # 构建表格数据
            table_data = [
                ['裂纹数量', f"{len(crack_analyses)} 处"],
                ['总面积', f"{total_area:,} 像素²"],
                ['最大宽度', f"{max_width:.1f} 像素"],
                ['总长度', f"{total_length:.0f} 像素"],
                ['严重程度', max(c['severity'] for c in crack_analyses)],
                ['检测时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]

            # 如果有实际尺寸
            if self.ratio and 'area_mm2' in crack_analyses[0]:
                table_data.append(['实际总面积', f"{sum(c['area_mm2'] for c in crack_analyses):.1f} mm²"])
                table_data.append(['实际最大宽度', f"{max_width * self.ratio:.1f} mm"])

            # 绘制表格
            table = axes[1, 1].table(cellText=table_data, colLabels=['指标', '数值'],
                                     loc='center', cellLoc='left', colWidths=[0.4, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.5)
            axes[1, 1].set_title('裂纹分析汇总', fontsize=12, pad=20)

            # 添加建议
            severity_list = [c['severity'] for c in crack_analyses]
            if '严重' in severity_list:
                suggestion_text = "建议：存在严重裂纹，需立即处理！"
            elif '中度' in severity_list:
                suggestion_text = "建议：存在中度裂纹，建议安排维修。"
            else:
                suggestion_text = "建议：裂纹程度轻微，持续观察即可。"

            axes[1, 1].text(0.5, 0.05, suggestion_text, transform=axes[1, 1].transAxes,
                            fontsize=11, ha='center', va='bottom', color='red', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, '未检测到裂纹\n\n建筑表面状况良好',
                            ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('分析结果', fontsize=12)

        plt.tight_layout()

        # 保存报告
        if save_path is None:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = f"{base_name}_report.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        print(f"报告已保存: {save_path}")
        print(f"检测结果: 发现 {len(crack_analyses)} 处裂纹")

        # 同时保存 JSON 数据
        if crack_analyses:
            json_path = save_path.replace('.png', '.json')
            output_data = {
                "image_name": os.path.basename(img_path),
                "detection_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_cracks": len(crack_analyses),
                "cracks": crack_analyses,
                "summary": {
                    "total_area_pixel": total_area,
                    "max_width_pixel": max_width,
                    "longest_length_pixel": total_length,
                    "most_severe": max(c['severity'] for c in crack_analyses)
                }
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"数据已保存: {json_path}")

        return crack_analyses, save_path


# 使用示例
if __name__ == '__main__':
    analyzer = CrackAnalyzer(
        model_path='D:/code/yolov11/runs/crack_optimized/finetune_v1/weights/best.pt',
        pixel_to_mm_ratio=None
    )

    image_path = 'D:/嵌入式设计/微信图片_20260516194116_161_454.jpg'
    results, report_path = analyzer.generate_report(image_path)

    for crack in results:
        print(f"\n裂纹 {crack['id']}:")
        print(f"   面积: {crack['area_pixel']:,} 像素²")
        print(f"   最大宽度: {crack['max_width_pixel']:.1f} 像素")
        print(f"   最长长度: {crack['longest_length_pixel']:.0f} 像素")
        print(f"   严重等级: {crack['severity']}")
        print(f"   建议: {crack['suggestion']}")
