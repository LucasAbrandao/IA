#include <bits/stdc++.h>
using namespace std;

// BFS - Breadth-First Search
void bfs(int start, const vector<vector<int>>& adjList) {
    vector<bool> visited(adjList.size(), false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        cout << current << " ";

        for (auto neighbor : adjList[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

int main() {
    // Grafo representado por lista de adjacência
    vector<vector<int>> adjList = {
        {1, 2},    // 0
        {0, 3, 4}, // 1
        {0, 5},    // 2
        {1},       // 3
        {1, 6, 7}, // 4
        {2, 8},    // 5
        {4},       // 6
        {4, 9},    // 7
        {5},       // 8
        {7}        // 9
    };

    int startVertex = 0;

    cout << "BFS: ";
    bfs(startVertex, adjList);
    cout << endl;

    return 0;
}
