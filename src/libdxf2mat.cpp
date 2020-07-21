#include "libdxf2mat.h"
#include "tinysplinecpp.h"
#include "dl_dxf.h"
#include "dl_creationadapter.h"

#include <cmath>
#include <vector>
#include <random>

namespace libdxf2mat {

	using namespace std;
	using namespace cv;

	template<typename T>
	static inline Point round_point(const Point_<T>& p)
	{
		return Point(round(p.x), round(p.y));
	}

	static double getRandomDoule()
	{
		static default_random_engine e;
		static uniform_real<double> distri(0.0, 1.0);
		return distri(e);
	}

	// 2D transformation
	class Trans2D {
	public:
		// zoom -> rotate -> translate
		Trans2D(double rad, double dx, double dy, double zoomx = 1.0, double zoomy = 1.0):
			mat{zoomx * cos(rad), -zoomy * sin(rad), dx, 
			    zoomx * sin(rad), zoomy * cos(rad), dy } 
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
			if (this != std::addressof(lh))
			{
				mat[0] = lh.mat[0];
				mat[1] = lh.mat[1];
				mat[2] = lh.mat[2];
				mat[3] = lh.mat[3];
				mat[4] = lh.mat[4];
				mat[5] = lh.mat[5];
			}
			return *this;
		}

		Trans2D& operator*=(const Trans2D& rh)
		{
			double m0 = mat[0] * rh.mat[0] + mat[1] * rh.mat[3];
			double m1 = mat[0] * rh.mat[1] + mat[1] * rh.mat[4];
			double m2 = mat[0] * rh.mat[2] + mat[1] * rh.mat[5] + mat[2];

			double m3 = mat[3] * rh.mat[0] + mat[4] * rh.mat[3];
			double m4 = mat[3] * rh.mat[1] + mat[4] * rh.mat[4];
			double m5 = mat[3] * rh.mat[2] + mat[4] * rh.mat[5] + mat[5];

			mat[0] = m0, mat[1] = m1, mat[2] = m2;
			mat[3] = m3, mat[4] = m4, mat[5] = m5;
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

		double mat[6] = { 1.0, 0.0, 0.0, 
			              0.0 ,1.0, 0.0 };
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
			if (this != std::addressof(lh))
			{
				pts = lh.pts;
				isClosed = lh.isClosed;
			}
			return *this;
		}
		RasterizeData& operator= (RasterizeData&& rh) noexcept
		{
			if (this != std::addressof(rh))
			{
				pts = std::move(rh.pts);
				isClosed = std::move(rh.isClosed);
			}
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

	class Ellipse {
	public:
		Ellipse(const Point2d& cen, const Size2d& axes, double rad, double beg, double End):
			m_cen(cen), m_axes(axes), m_rot_rad(rad), m_beg(beg), m_end(End)
		{}

		RasterizeData discretize(double interval, const Trans2D& trans) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = max(m_axes.width, m_axes.height) * zoom;
			int pt_nums = max(1, int(ceil((m_end - m_beg) / interval))) + 1;
			double step = (m_end - m_beg) / (pt_nums - 1);

			double theta = m_beg;
			RasterizeData ras_data;
			ras_data.isClosed = false;
			// Note: tran2d and trans have the same determinant.
			Trans2D tran2d = trans * Trans2D(m_rot_rad, m_cen.x, m_cen.y);
			for (int i = 0; i < pt_nums; ++i)
			{
				Point2d pt(cos(theta) * m_axes.width, sin(theta) * m_axes.height);
				ras_data.pts.emplace_back(round_point(tran2d.transfer(pt)));
				theta += step;
			}
			return ras_data;			
		}

		Point2d m_cen;
		Size2d m_axes;
		double m_rot_rad;
		double m_beg;
		double m_end;
	};

	// A pipline between 2D nurbs to 3D b-spline
	class tinynurbs2d :public tinyspline::BSpline {

	public:
		tinynurbs2d(size_t nCtrlp, size_t deg = 3, tinyspline::BSpline::type type = TS_CLAMPED) :
			BSpline(nCtrlp, 3, deg, type) {}

		Point2d at(double u) const
		{
			auto val = BSpline::eval(u).result();
			return Point2d(val[0], val[1]);
		}

		void setNurbsControlPoints(const vector<Vec3d>& conPs)
		{
			vector<tinyspline::real> pts;
			for (const auto& p : conPs)
			{
				pts.push_back(p[0] * p[2]);
				pts.push_back(p[1] * p[2]);
				pts.push_back(p[2]);
			}
			BSpline::setControlPoints(pts);
		}

		vector<Point2d> sampleNurbs(size_t num) const
		{
			auto wPts = BSpline::sample(num);
			vector<Point2d> pts;
			for (int i = 0; i < wPts.size() - 2; i += 3)
			{
				double w = wPts[i + 2];
				if (w == 0.0)
					continue;

				pts.emplace_back(wPts[i] / w, wPts[i + 1] / w);
			}
			return pts;
		}

	private:
		using BSpline::setControlPoints;
		using BSpline::sample;

	};

	class Spline {
	public:
		Spline(unsigned _degree, const vector<Vec3d>& vs, const vector<double>& _knots, int _flag, size_t _sampleNum) :
			degree(_degree), vertexes(vs), knots(_knots), flag(_flag),
			m_impl(vertexes.size(), degree)
		{
			m_impl.setNurbsControlPoints(vertexes);
			vector<tinyspline::real> ks;
			for (const auto& kn : knots)
				ks.push_back(kn);
			m_impl.setKnots(ks);
		}

		RasterizeData discretize(double interval, const Trans2D& trans) const
		{
			// roughly estimates sample points number
			double dt = min(0.002, 2.0 / vertexes.size());
			unsigned long long pt_nums = max(1000ull, vertexes.size());
			double range = 1.0 - dt;
			double samp = dt / 2.0;
			double len = 0.0;
			for (int i = 0; i < 10; ++i)
			{
				double pos = samp + getRandomDoule() * range;
				auto val_left = trans.transfer(m_impl.at(pos - samp));
				auto val_right = trans.transfer(m_impl.at(pos + samp));
				len += sqrt(pow(val_left.x - val_right.x, 2) + pow(val_left.y - val_right.y, 2));
			}
			len /= (10.0 * dt);
			int pt_nums = max(1, int(min(1000000.0, ceil(len / interval)))) + 1;

			auto pts = m_impl.sampleNurbs(pt_nums);
			RasterizeData ras_data;
			ras_data.isClosed = flag;
			for (const auto& p : pts)
				ras_data.pts.emplace_back(round_point(trans.transfer(p)));
			return ras_data;
		}

		unsigned int degree = 2;
		vector<Vec3d> vertexes;
		vector<double> knots;
		int flag = false;

	private:
		tinynurbs2d m_impl;
	};



}  // namespace libdxf2mat


