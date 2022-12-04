#pragma once

#include "Scene.h"
#include "Viewport.h"
#include "AutoMorphingModel.h"
#include <utility>

class BasicScene : public cg3d::Scene
{
public:
    explicit BasicScene(std::string name, cg3d::Display *display) : Scene(std::move(name), display){};
    void Init(float fov, int width, int height, float near, float far);
    void Update(const cg3d::Program &program, const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view, const Eigen::Matrix4f &model) override;
    void KeyCallback(cg3d::Viewport *viewport, int x, int y, int key, int scancode, int action, int mods) override;
    static void best_edge_and_point(
            const int e,
            const Eigen::MatrixXd & V,
            const Eigen::MatrixXi & /*F*/,
            const Eigen::MatrixXi & E,
            const Eigen::VectorXi & /*EMAP*/,
            const Eigen::MatrixXi & /*EF*/,
            const Eigen::MatrixXi & /*EI*/,
            double & cost,
            Eigen::RowVectorXd & p);

private:
    bool collapseTenPerEdges(float ratio);
    std::shared_ptr<Movable> root;
    std::shared_ptr<cg3d::Model> obj;
};
