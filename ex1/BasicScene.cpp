//assignment1
//Roni Amos 318286275
//Oren Biezuner 207468901

#include "./BasicScene.h"
#include <read_triangle_mesh.h>
#include <utility>
#include "ObjLoader.h"
#include "SceneWithImGui.h"
#include "CamModel.h"
#include "Visitor.h"
#include "Utility.h"
#include "IglMeshLoader.h"
#include "igl/min_heap.h"
#include "igl/read_triangle_mesh.cpp"
#include "igl/edge_flaps.h"
#include "igl/parallel_for.h"
#include "igl/shortest_edge_and_midpoint.h"
#include "igl/collapse_edge.h"
#include "igl/per_vertex_normals.h"
#include <memory>
#include "igl/per_face_normals.h"

using namespace cg3d;

std::shared_ptr<cg3d::AutoMorphingModel> autoObj;
int meshDataIndex = 0;

//////hash function====================================================
template<typename T>
struct matrix_hash  {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (size_t i = 0; i < matrix.size(); ++i) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::unordered_map<Eigen::RowVector3d, Eigen::Matrix4d, matrix_hash<Eigen::RowVector3d>> map;
//////==================================================================

Eigen::VectorXi EMAP;
Eigen::MatrixXi OF, F, E, EF, EI;
Eigen::VectorXi EQ;
Eigen::MatrixXd OV;
igl::min_heap<std::tuple<double, int, int>> Q; // cost, Edge_index, new_Vertex_index
// If an edge were collapsed, we'd collapse it to these points:
Eigen::MatrixXd V, C;
int num_collapsed;

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    /// create camera
    camera = Camera::Create("camera", fov, float(width) / height, near, far);

    /// create backround
    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{std::make_shared<Material>("daylight", "shaders/cubemapShader")};
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{Model::Create("background", Mesh::Cube(), daylight)};
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();

    auto program = std::make_shared<Program>("shaders/basicShader");
    auto material{std::make_shared<Material>("material", program)}; // empty material
                                                                                        // SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());
    ///create object
    material->AddTexture(0, "textures/box0.bmp", 2);
    auto objMesh{IglLoader::MeshFromFiles("cube_igl", "data/cow.off")};

    obj = cg3d::Model::Create("obj", objMesh, material);
    auto morphFunc = [](cg3d::Model *model, cg3d::Visitor *visitor)
    {
        return meshDataIndex;
    };

    autoObj = cg3d::AutoMorphingModel::Create(*obj, morphFunc);

    autoObj->Translate({0, 0, 0});
    autoObj->Scale(8.0f);
    autoObj->showWireframe = true;
    camera->Translate(20, Axis::Z);
    root->AddChild(autoObj);

    auto mesh = autoObj->GetMeshList();

    /// Function to reset original mesh and data structures
    OV = mesh[0]->data[0].vertices;
    OF = mesh[0]->data[0].faces;

    F = OF;
    V = OV;

    /// getting normals per faces
    Eigen::MatrixXd N;
    igl::per_face_normals(V, F, N);

    igl::edge_flaps(F, E, EMAP, EF, EI);
    C.resize(E.rows(), V.cols());
    Eigen::VectorXd costs(E.rows());

    Q = {};
    EQ = Eigen::VectorXi::Zero(E.rows());
    {
        igl::parallel_for(F.rows(), [&](const int f)
        {
            /// iterate over faces, each face get edges, get vertices.
            /// for each vertex go to map and add face normal (Kp) to it's Q.
            auto v1 = V.row(F(f, 0));
            auto v2 = V.row(F(f, 1));
            auto v3 = V.row(F(f, 2));

            auto p = N.row(f);

            auto d1 = -(p.x() * v1.x() + p.y() * v1.y() + p.z() * v1.z());
            auto d2 = -(p.x() * v2.x() + p.y() * v2.y() + p.z() * v2.z());
            auto d3 = -(p.x() * v3.x() + p.y() * v3.y() + p.z() * v3.z());

            Eigen::MatrixXd m1(4,4);
            m1 << p.x()*p.x(), p.x()*p.y(), p.x()*p.z(), p.x()*d1,
                p.x()*p.y(), p.y()*p.y(), p.y()*p.z(), p.y()*d1,
                p.x()*p.z(), p.y()*p.z(), p.z()*p.z(), p.z()*d1,
                p.x()*d1, p.y()*d1, p.z()*d1, d1*d1;

            Eigen::MatrixXd m2(4,4);
            m2 << p.x()*p.x(), p.x()*p.y(), p.x()*p.z(), p.x()*d2,
                    p.x()*p.y(), p.y()*p.y(), p.y()*p.z(), p.y()*d2,
                    p.x()*p.z(), p.y()*p.z(), p.z()*p.z(), p.z()*d2,
                    p.x()*d2, p.y()*d2, p.z()*d2, d2*d2;

            Eigen::MatrixXd m3(4,4);
            m3 << p.x()*p.x(), p.x()*p.y(), p.x()*p.z(), p.x()*d3,
                    p.x()*p.y(), p.y()*p.y(), p.y()*p.z(), p.y()*d3,
                    p.x()*p.z(), p.y()*p.z(), p.z()*p.z(), p.z()*d3,
                    p.x()*d3, p.y()*d3, p.z()*d3, d3*d3;

            map[v1] += m1;
            map[v2] += m2;
            map[v3] += m3;
        }, 10000);

        Eigen::VectorXd costs(E.rows());
        igl::parallel_for(
            E.rows(), [&](const int e)
            {
                /// iterate over the map, for each vertex, calaulates cost and add to priority queue.
                double cost = e;
                Eigen::RowVectorXd p(1,3);
                best_edge_and_point(e,V,F,E,EMAP,EF,EI,cost,p);
                C.row(e) = p;
                costs(e) = cost; },
            10000);
        for (int e = 0; e < E.rows(); e++)
        {
            Q.emplace(costs(e), e, 0);
        }
    }

    num_collapsed = 0;

    // std::cout<< "vertices: \n" << V <<std::endl;
    // std::cout<< "faces: \n" << F <<std::endl;

    // std::cout<< "edges: \n" << E.transpose() <<std::endl;
    // std::cout<< "edges to faces: \n" << EF.transpose() <<std::endl;
    // std::cout<< "faces to edges: \n "<< EMAP.transpose()<<std::endl;
    // std::cout<< "edges indices: \n" << EI.transpose() <<std::endl;
}

/**
 * @param ratio float between 0 to 1. e.g 0.01 will remove 10 percent of the edges
 */
bool BasicScene::collapseTenPerEdges(float ratio)
{
    std::cout << "Start collapsing:" << std::endl;
    static int numOfCollapses = 0;
    numOfCollapses++;
    /// If animating then collapse 10% of edges
    if (!Q.empty())
    {
        bool something_collapsed = false;
        /// collapse edge
        const int max_iter = std::ceil(ratio * Q.size());
        for (int j = 0; j < max_iter; j++)
        {
            double cost = std::get<0>(Q.top());
            if (!igl::collapse_edge(igl::shortest_edge_and_midpoint, V, F, E, EMAP, EF, EI, Q, EQ, C))
            {
                break;
            }
            something_collapsed = true;
            num_collapsed++;
        }
        /// if collapsed, insert to mesh list
        if (something_collapsed)
        {
            Eigen::MatrixXd normals = Eigen::MatrixXd();
            igl::per_vertex_normals(V, F, normals);
            Eigen::MatrixXd texCoords = Eigen::MatrixXd::Zero(V.rows(), 2);
            auto newlist = autoObj->GetMeshList();
            newlist[0]->data.push_back({V, F, normals, texCoords});
            autoObj->SetMeshList(newlist);
        }
    }
    std::cout << "End collapsing:" << std::endl;
    return false;
}

int frameCount = 0;

void BasicScene::Update(const Program &program, const Eigen::Matrix4f &proj, const Eigen::Matrix4f &view, const Eigen::Matrix4f &model)
{
    frameCount++;
    Scene::Update(program, proj, view, model);
    program.SetUniform4f("lightColor", 1.0f, 1.0f, 1.0f, 0.5f);
    program.SetUniform4f("Kai", 1.0f, 1.0f, 1.0f, 1.0f);
//    autoObj->Rotate(0.005f, Axis::XYZ);
}

void BasicScene::KeyCallback(cg3d::Viewport *viewport, int x, int y, int key, int scancode, int action, int mods)
{
    auto system = camera->GetRotation().transpose();

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        switch (key) // NOLINT(hicpp-multiway-paths-covered)
        {
            case GLFW_KEY_SPACE:
                collapseTenPerEdges(0.01f);
                meshDataIndex = obj->GetMeshList()[0]->data.size() - 1;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_UP:
                if (meshDataIndex != obj->GetMeshList()[0]->data.size())
                {
                    meshDataIndex++;
                }
                // camera->RotateInSystem(system, 0.1f, Axis::X);
                break;
            case GLFW_KEY_DOWN:
                if (meshDataIndex != 0)
                {
                    meshDataIndex--;
                }
                // camera->RotateInSystem(system, -0.1f, Axis::X);
                break;
        }
    }
}

///=====================================================================================================
///algorithm
void BasicScene::best_edge_and_point(
        const int e,
        const Eigen::MatrixXd & V,
        const Eigen::MatrixXi & /*F*/,
        const Eigen::MatrixXi & E,
        const Eigen::VectorXi & /*EMAP*/,
        const Eigen::MatrixXi & /*EF*/,
        const Eigen::MatrixXi & /*EI*/,
        double & cost,
        Eigen::RowVectorXd & p)
{
    auto v1 = V.row(E(e, 0));
    auto v2 = V.row(E(e, 1));
    ///p1 the vertex to replace the edge- case 1: middle
    Eigen::RowVectorXd p1 = v1;
    p1 += v2;
    p1 *= 0.5;
    auto q1 = map[v1];
    auto q2 = map[v2];
    auto q = q1;
    q += q2;
    Eigen::MatrixXd vExtendedTranspose (1, 4);
    vExtendedTranspose << p1.transpose().x(), p1.transpose().y(), p1.transpose().z(), 1;
    Eigen::MatrixXd vExtended (4, 1);
    vExtended << p1.x(), p1.y(), p1.z(), 1;
    vExtendedTranspose *= q;
    vExtendedTranspose *= vExtended;
    auto cost1 = vExtendedTranspose(0, 0);

    auto v11 = V.row(E(e, 0));
    /// case 2 v1 replaces the edge
    Eigen::RowVectorXd p2 = v11;
    auto q11 = map[v11];
    Eigen::MatrixXd vExtendedTranspose1 (1, 4);
    vExtendedTranspose1 << p2.transpose().x(), p2.transpose().y(), p2.transpose().z(), 1;
    Eigen::MatrixXd vExtended1 (4, 1);
    vExtended1 << p2.x(), p2.y(), p2.z(), 1;
    vExtendedTranspose1 *= q11;
    vExtendedTranspose1 *= vExtended1;
    auto cost2 = vExtendedTranspose1(0, 0);

    auto v22 = V.row(E(e, 1));
    /// case 3 v2 replaces the edge
    Eigen::RowVectorXd p3 = v22;
    auto q22 = map[v22];
    Eigen::MatrixXd vExtendedTranspose2 (1, 4);
    vExtendedTranspose2 << p3.transpose().x(), p3.transpose().y(), p3.transpose().z(), 1;
    Eigen::MatrixXd vExtended2 (4, 1);
    vExtended2 << p3.x(), p3.y(), p3.z(), 1;
    vExtendedTranspose2 *= q22;
    vExtendedTranspose2 *= vExtended2;
    auto cost3 = vExtendedTranspose2(0, 0);

    cost = std::min(std::min(cost1, cost2), cost3);/// best cost (min cost)
    if (cost == cost1) p = p1;
    else if (cost == cost2) p = p2;
    else p = p3;
}
///=====================================================================================================