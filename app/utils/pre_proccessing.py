import cv2
import numpy as np

class ProccessingImage:
    def __init__(self):
        pass

    def canny(self, img, min_threshold, max_threshold):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(gray_img, min_threshold, max_threshold)

        return canny_img
    
    def get_lines(self, img_canny, threshold):
        lines = cv2.HoughLines(img_canny, 1, np.pi/180, threshold)

        return lines

    def line_intersection(self, line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])
        x0, y0 = np.linalg.lstsq(A, b, rcond=None)[0]
        return [int(np.round(x0)), int(np.round(y0))]

    def rotate_image(self, img):
        height, width = img.shape[:2]
        center = (width//2, height//2)
        canny = cv2.Canny(img, 50, 200)
        lines = cv2.HoughLines(canny, 1, np.pi/180, 180)
        min_horizontal_rad = np.deg2rad(45)
        max_horizontal_rad = np.deg2rad(135)

        for line in lines:
            rho, theta = line[0]
            if min_horizontal_rad < theta < max_horizontal_rad:
                horizontal_theta = theta
                break

        angle_rad = horizontal_theta - (np.pi/2)
        angle_deg = np.rad2deg(angle_rad)
        M = cv2.getRotationMatrix2D(center=center, angle=angle_deg, scale=1)

        rotated_image = cv2.warpAffine(
            src=img.copy(),
            M=M, 
            dsize=(width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated_image
    
    def focus_image(self, img_path):
        horizontal_lines = []
        vertical_lines = []

        angle_threshold_rad = np.deg2rad(20)
        horizontal_target_rad = np.pi / 2
        img = cv2.imread(img_path)

        rotated_image = self.rotate_image(img)
        canny = cv2.Canny(rotated_image, 50, 150)
        lines = cv2.HoughLines(canny, 1, np.pi/180, 180)

        # return rotated_image
        for line in lines:
            rho, theta = line[0]

            # Kiểm tra đường ngang (gần 90 độ)
            if abs(theta - horizontal_target_rad) < angle_threshold_rad:
                horizontal_lines.append((rho, theta))
            
            # Kiểm tra đường dọc (gần 0 hoặc 180 độ)
            elif abs(theta) < angle_threshold_rad or abs(theta - np.pi) < angle_threshold_rad:
                vertical_lines.append((rho, theta))

        if not horizontal_lines or not vertical_lines:
            print("Không tìm đủ đường ngang hoặc dọc, hãy thử điều chỉnh ngưỡng Canny hoặc HoughLines.")
        else:
            horizontal_lines.sort(key=lambda x: x[0])
            vertical_lines.sort(key=lambda x: x[0])

        # Lấy đường có rho nhỏ nhất và lớn nhất
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        left_line = vertical_lines[-1]
        right_line = vertical_lines[0]

        corner_tl = self.line_intersection(top_line, left_line)
        corner_tr = self.line_intersection(top_line, right_line)
        corner_bl = self.line_intersection(bottom_line, left_line)
        corner_br = self.line_intersection(bottom_line, right_line)
        src_points = np.float32([corner_tl, corner_tr, corner_br, corner_bl])

        # Tính chiều rộng và chiều cao mới
        # Chiều rộng = khoảng cách trung bình giữa (tl, tr) và (bl, br)
        width_top = np.linalg.norm(np.array(corner_tl) - np.array(corner_tr))
        width_bottom = np.linalg.norm(np.array(corner_bl) - np.array(corner_br))
        max_width = int(max(width_top, width_bottom))
        # Chiều cao = khoảng cách trung bình giữa (tl, bl) và (tr, br)
        height_left = np.linalg.norm(np.array(corner_tl) - np.array(corner_bl))
        height_right = np.linalg.norm(np.array(corner_tr) - np.array(corner_br))
        max_height = int(max(height_left, height_right))

        # Tọa độ đích (hình chữ nhật hoàn hảo)
        dst_points = np.float32([
            [0, 0],              # Top-left
            [max_width - 1, 0],  # Top-right
            [max_width - 1, max_height - 1], # Bottom-right
            [0, max_height - 1]  # Bottom-left
        ])

        # Lấy ma trận biến đổi
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Áp dụng biến đổi
        cropped_image = cv2.warpPerspective(rotated_image, M, (max_width, max_height))

        return cropped_image

if __name__ == "__main__":
    image = cv2.imread(r"temp\temp.jpg")
    processingImage = ProccessingImage()
    cropped_image = processingImage.focus_image(r"temp\temp.jpg")