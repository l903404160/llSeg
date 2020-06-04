import numpy as np
import PIL.Image as Image
from .base import Transform


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.
    The fill color is used to map pixels from the source rect that fall outside
    the source image.
    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class PadCropTransform(Transform):
    def __init__(self, pad_w: int, pad_h: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        p_h = int(self.pad_h // 2) + 1
        p_w = int(self.pad_w // 2) + 1
        if len(img.shape) < 3:
            img = np.pad(img, ((p_h, p_h),(p_w, p_w)), 'constant', constant_values=255)
            return img[0 : self.h, 0: self.w]
        elif len(img.shape) == 3:
            img = np.pad(img, ((p_h, p_h),(p_w, p_w), (0,0)), 'constant', constant_values=0)
            return img[0 : self.h, 0 : self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        # TODO Pad coords
        """
        Apply crop transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        # TODO Pad polygons
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(
            self.x0, self.y0, self.x0 + self.w, self.y0 + self.h
        ).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]

