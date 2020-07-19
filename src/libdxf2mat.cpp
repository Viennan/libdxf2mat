#include "libdxf2mat.h"
#include "tinysplinecpp.h"
#include "dl_dxf.h"
#include "dl_creationadapter.h"

#include <cmath>
#include <vector>

namespace libdxf2mat {

	using namespace std;
	using namespace cv;

	template<typename T>
	static inline Point round_point(const Point_<T>& p)
	{
		return Point(round(p.x), round(p.y));
	}

	// 2D transformation
	class Trans2D {
	public:
		// zoom -> rotate -> translate
		Trans2D(double rad, double dx, double dy, double zoomx, double zoomy):
			mat{zoomx * cos(rad), -zoomy * sin(rad), dx, 
			    zoomx * sin(rad), zoomy * cos(rad), dy} 
		{}

		Trans2D(const Trans2D& trans) :
			mat{ trans.mat[0], trans.mat[1], trans.mat[2], 
			     trans.mat[3], trans.mat[4], trans.mat[5] } 
		{}

		Trans2D(double zoomx, double zoomy):
			mat{zoomx, 0.0, 0.0,
		        0.0, zoomy, 0.0}
		{}

		Trans2D& operator= (const Trans2D& lh) 
		{
			memcpy(mat, lh.mat, sizeof(mat));
			return *this;
		}

		Trans2D& operator*=(const Trans2D& rh)
		{
			for (int i = 0; i < 2; ++i)
			{
				int base = 3 * i;
				double m0 = mat[base], m1 = mat[base+1], m2 = mat[base+2];
				for (int j = 0; j < 3; ++j)
					mat[base + j] = m0 * rh.mat[j] + m1 * rh.mat[j + 3] + m2 * (j >> 1);
			}
			return *this;
		}

		template<typename T>
		Point_<T> transfer(const Point_<T>& p) const
		{
			return Point_<T>(mat[0] * p.x + mat[1] * p.y + mat[2],
				mat[3] * p.x + mat[4] * p.y + mat[5]);
		}

		template<>
		Point transfer(const Point& p) const
		{
			return round_point(transfer(Point2d(p.x, p.y)));
		}

		double det() const
		{
			return mat[0] * mat[4] - mat[1] * mat[3];
		}

		double mat[6] = { 1.0, 0.0, 0.0, 0.0 ,1.0, 0.0 };
	};

	static inline Trans2D operator*(const Trans2D& lh, const Trans2D& rh)
	{
		Trans2D tmp(lh);
		return tmp *= rh;
	}

	struct RasterizeData {
		RasterizeData(): pts(), isClosed(false) 
		{};
		RasterizeData(const vector<Point>& ps, bool flag) :
			pts(ps), isClosed(flag)
		{};
		RasterizeData(vector<Point>&& ps, bool flag) :
			pts(std::move(ps)), isClosed(flag)
		{};
		RasterizeData(const RasterizeData& lh) :
			pts(lh.pts), isClosed(lh.isClosed)
		{};
		RasterizeData(RasterizeData&& rh) noexcept:
		pts(std::move(rh.pts)), isClosed(std::move(rh.isClosed))
		{};
		RasterizeData& operator= (const RasterizeData& lh)
		{
			pts = lh.pts;
			isClosed = lh.isClosed;
			return *this;
		}
		RasterizeData& operator= (RasterizeData&& rh) noexcept
		{
			pts = std::move(rh.pts);
			isClosed = std::move(rh.isClosed);
			return *this;
		}

		vector<Point> pts;
		bool isClosed;
	};

	class Line {
	public:
		Line(const Point2d& from, const Point2d& to):
			m_pt1(from), m_pt2(to)
		{}

		RasterizeData discretize(double interval, const Trans2D& trans) const
		{
			RasterizeData ras_data;
			ras_data.isClosed = false;
			ras_data.pts.emplace_back(round_point(trans.transfer(m_pt1)));
			ras_data.pts.emplace_back(round_point(trans.transfer(m_pt2)));
			return ras_data;
		}
	    
		Point2d m_pt1;
		Point2d m_pt2;
	};

	class Circle {
	public:
		Circle(const Point2d& cen, double r) : m_cen(cen), m_r(r) {}
		Circle(double x, double y, double r) : m_cen(x, y), m_r(r) {}

		RasterizeData discretize(double interval, const Trans2D& trans) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = zoom * m_r;
			int pt_nums = max(1, int(ceil(2.0 * CV_PI * r / interval)));
			double step = 2 * CV_PI / pt_nums;

			double theta = 0.0;
			RasterizeData ras_data;
			ras_data.isClosed = true;
			for (int i = 0; i < pt_nums; ++i)
			{
				Point2d pt(m_r * cos(theta), m_r * sin(theta));
				pt += m_cen;
				ras_data.pts.emplace_back(round_point(trans.transfer(pt)));
				theta += step;
			}
			return ras_data;
		}

		Point2d m_cen;
		double m_r;
	};

	class Arc {
	public:
		Arc(const Point2d& cen, double r, double rad_beg, double rad_end):
			m_cen(cen), m_r(r), m_beg(rad_beg), m_end(rad_end)
		{}

		RasterizeData discretize(double interval, const Trans2D& trans) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = zoom * m_r;
			int pt_nums = max(1, int(ceil(r * (m_end - m_beg) / interval))) + 1;
			double step = (m_end - m_beg) / (pt_nums - 1);

			double theta = m_beg;
			RasterizeData ras_data;
			ras_data.isClosed = false;
			for (int i = 0; i < pt_nums; ++i)
			{
				Point2d pt(m_r * cos(theta), m_r * sin(theta));
				pt += m_cen;
				ras_data.pts.emplace_back(round_point(trans.transfer(pt)));
				theta += step;
			}
			return ras_data;
		}
	
		Point2d m_cen;
		double m_r;
		double m_beg;
		double m_end;
	};

}  // namespace libdxf2mat


