#include "libdxf2mat.h"
#include "tinysplinecpp.h"
#include "dl_dxf.h"
#include "dl_creationadapter.h"

#include <cmath>
#include <vector>
#include <algorithm>

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
			mat[0] = lh.mat[0];
			mat[1] = lh.mat[1];
			mat[2] = lh.mat[2];
			mat[3] = lh.mat[3];
			mat[4] = lh.mat[4];
			mat[5] = lh.mat[5];
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

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
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

	// Use Micke's Formula to calculate center with begin point and end point and bulge for all cases.
	static inline double mickeFormula(const Vec3d& b, const Vec3d& e, Point2d& cen)
	{
		auto bulge = 0.5 * (1.0 / b[2] - b[2]);
		cen.x = 0.5 * ((b[0] + e[0]) - bulge * (e[1] - b[1]));
		cen.y = 0.5 * ((b[1] + e[1]) + bulge * (e[0] - b[0]));
		return sqrt(pow(b[0] - cen.x, 2) + pow(b[1] - cen.y, 2));
	}

	class Polylines {
	public:
		Polylines(const vector<Vec3d>& vs, bool closed):
			vertices(vs), isClosed(closed), m_len(0.0), isOpt(false)
		{
			// If it is a optimize polylines
			for (const auto& v : vs)
			{
				if (v[2] != 0.0)
				{
					isOpt = true;
					break;
				}
			}
			if (isOpt)
			{
				// For optimize polylines
				size_t n = vertices.size();
				auto n_minus_1 = n - 1;
				for (size_t i = 0; i < n_minus_1; ++i)
				{
					Point2d p;
					auto r = mickeFormula(vertices[i], vertices[i + 1], p);
					m_len += 4.0 * r * atan(vertices[i][2]);
				}
				if (isClosed)
				{
					Point2d p;
					auto r = mickeFormula(vertices[n - 1], vertices[0], p);
					m_len = 4.0 * r * atan(vertices[n - 1][2]);
				}
			}
			else
			{
				// For normal polylines
				size_t n = vertices.size();
				auto n_minus_1 = n - 1;
				for (size_t i = 0; i < n_minus_1; ++i)
				{
					m_len += sqrt(pow(vertices[i][0] - vertices[i + 1][0], 2) + 
						pow(vertices[i][1] - vertices[i + 1][1], 2));
				}
				if (isClosed)
					m_len += sqrt(pow(vertices[n - 1][0] - vertices[0][0], 2) +
						pow(vertices[n - 1][1] - vertices[0][1], 2));
			}
		}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			RasterizeData ras_data;
			ras_data.isClosed = isClosed;
			int freepts = max_pts < vertices.size() * 2 ? 0 : (max_pts - vertices.size());
			if (isOpt && freepts > 0)
			{
				freepts = ceil(min(double(freepts), m_len * sqrt(trans.det()) / interval));
				size_t n = vertices.size();
				for (size_t i = 0; i < n; ++i)
				{
					const auto& vb = vertices[i];
					ras_data.pts.emplace_back(round_point(trans.transfer(Point2d(vb[0], vb[1]))));
					if (!isClosed && i == n - 1)
						break;
					const auto& ve = (i == n - 1) ? vertices[0] : vertices[i + 1];
					if (vb[2] != 0.0)
					{
						Point2d cen;
						auto r = mickeFormula(vb, ve, cen);
						auto cen_rad = 4.0 * atan(vb[2]);
						auto len = r * cen_rad;
						int pt_nums = round(len / m_len * freepts);
						if (pt_nums > 0)
						{
							double step = cen_rad / (pt_nums + 1);
							double theta = acos((vb[0] - cen.x) / r);
							theta = vb[1] < cen.y ? (2.0 * CV_PI - theta) : theta;
							theta += step;
							for (int j = 0; j < pt_nums; ++j)
							{
								Point2d p(r * cos(theta), r * sin(theta));
								p += cen;
								ras_data.pts.emplace_back(round_point(trans.transfer(p)));
								theta += step;
							}
						}
					}
				}
			}
			else
			{
				ras_data.pts.resize(vertices.size());
				std::transform(vertices.cbegin(), vertices.cend(), ras_data.pts.begin(),
					[&trans](const Vec3d& p) {
						return round_point(trans.transfer(Point2d(p[0], p[1])));
					});
			}
			return ras_data;
		}

		vector<Vec3d> vertices;
		bool isClosed;
	private:
		double m_len;
		bool isOpt;
	};

	class Circle {
	public:
		Circle(const Point2d& cen, double r) : m_cen(cen), m_r(r) {}
		Circle(double x, double y, double r) : m_cen(x, y), m_r(r) {}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = zoom * m_r;
			size_t pt_nums = max(1ull, size_t(min(double(max_pts - 1), ceil(2.0 * CV_PI * r / interval))));
			double step = 2 * CV_PI / pt_nums;

			double theta = 0.0;
			RasterizeData ras_data;
			ras_data.isClosed = true;
			ras_data.pts.resize(pt_nums);
			for (size_t i = 0; i < pt_nums; ++i)
			{
				Point2d pt(m_r * cos(theta), m_r * sin(theta));
				pt += m_cen;
				ras_data.pts[i] = round_point(trans.transfer(pt));
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

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = zoom * m_r;
			size_t pt_nums = max(2ull, size_t(min(double(max_pts - 1), ceil(r * (m_end - m_beg) / interval))) + 1ull);
			double step = (m_end - m_beg) / (pt_nums - 1);

			double theta = m_beg;
			RasterizeData ras_data;
			ras_data.isClosed = false;
			ras_data.pts.resize(pt_nums);
			for (size_t i = 0; i < pt_nums; ++i)
			{
				Point2d pt(m_r * cos(theta), m_r * sin(theta));
				pt += m_cen;
				ras_data.pts[i] = round_point(trans.transfer(pt));
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

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double zoom = sqrt(trans.det());
			double r = max(m_axes.width, m_axes.height) * zoom;
			size_t pt_nums = max(2ull, size_t(min(double(max_pts - 1), ceil(r * (m_end - m_beg) / interval))) + 1ull);
			double step = (m_end - m_beg) / (pt_nums - 1);

			double theta = m_beg;
			RasterizeData ras_data;
			ras_data.isClosed = false;
			ras_data.pts.resize(pt_nums);
			// Note: tran2d and trans have the same determinant.
			Trans2D tran2d = trans * Trans2D(m_rot_rad, m_cen.x, m_cen.y);
			for (int i = 0; i < pt_nums; ++i)
			{
				Point2d pt(cos(theta) * m_axes.width, sin(theta) * m_axes.height);
				ras_data.pts[i] = round_point(tran2d.transfer(pt));
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
		Spline(unsigned _degree, const vector<Vec3d>& vs, const vector<double>& _knots, 
			int closed, size_t _sampleNum) :
			degree(_degree), vertexes(vs), knots(_knots), isClosed(closed), 
			m_impl(vertexes.size(), degree), m_length(0.0)
		{
			m_impl.setNurbsControlPoints(vertexes);
			vector<tinyspline::real> ks;
			for (const auto& kn : knots)
				ks.push_back(kn);
			m_impl.setKnots(ks);

			// roughly estimates the length of spline
			auto pts = m_impl.sampleNurbs(100);
			int num = pts.size();
			for (size_t i = 1; i < num; ++i)
			{
				double dis = sqrt(pow(pts[i].x - pts[i - 1].x, 2) + pow(pts[i].y - pts[i - 1].y, 2));
				m_length += dis;
			}
			if (isClosed)
			{
				double dis = sqrt(pow(pts[num - 1].x - pts[0].x, 2) + pow(pts[num - 1].y - pts[0].y, 2));
				m_length += dis;
			}
		}

		RasterizeData discretize(double interval, const Trans2D& trans, int max_pts) const
		{
			// roughly estimates sample points number
			double len = m_length * sqrt(trans.det());
			size_t pt_nums = max(2ull, size_t(min(double(max_pts - 1), 
				ceil(len / interval))) + 1ull);
			auto pts = m_impl.sampleNurbs(pt_nums);
			RasterizeData ras_data;
			ras_data.isClosed = isClosed;
			ras_data.pts.resize(pts.size());
			std::transform(pts.cbegin(), pts.cend(), ras_data.pts.begin(),
				[&trans](const Point2d& p) { 
					return round_point(trans.transfer(p)); 
				});
			return ras_data;
		}

		unsigned int degree;
		vector<Vec3d> vertexes;
		vector<double> knots;
		bool isClosed;

	private:
		tinynurbs2d m_impl;
		double m_length;
	};



}  // namespace libdxf2mat


